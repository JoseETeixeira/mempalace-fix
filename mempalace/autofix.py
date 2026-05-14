"""Safe, idempotent palace auto-fixes.

Status and other read-mostly commands invoke :func:`auto_fix_palace` so the
operator does not have to run individual repair sub-commands by hand when
the fix is fast, well-understood, and reversible (we always take a sqlite
backup before mutating). Heavy rebuilds (full HNSW reindex) are deliberately
out of scope — those still go through ``mempalace repair --mode legacy``.

Each fixer is:

* **Detect-first.** A read-only probe decides whether to mutate. Probes
  must never raise on healthy state.
* **Idempotent.** Running twice in a row is a no-op the second time.
* **Bounded.** Each fixer touches a small, well-typed set of rows. No
  full-table rewrites, no schema changes.
* **Self-reporting.** Returns a structured ``FixReport`` describing what
  it did so the caller can show the operator.

The auto-fix entry point itself never raises — a fix that fails is logged
and reported, but does not block the rest of the status call.
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class FixReport:
    name: str
    detected: bool = False
    applied: bool = False
    rows_affected: int = 0
    detail: str = ""
    error: Optional[str] = None


@dataclass
class AutoFixSummary:
    palace_path: str
    backup_path: Optional[str] = None
    reports: list[FixReport] = field(default_factory=list)

    @property
    def any_applied(self) -> bool:
        return any(r.applied for r in self.reports)

    @property
    def any_detected(self) -> bool:
        return any(r.detected for r in self.reports)


def auto_fix_palace(
    palace_path: str,
    *,
    enabled: bool = True,
    backup: bool = True,
    apply: bool = True,
) -> AutoFixSummary:
    """Run every safe auto-fix against ``palace_path``.

    Args:
        enabled: Master switch. When False, returns an empty summary
            without touching disk. Lets callers respect ``--no-autofix``.
        backup: Take a single ``chroma.sqlite3.autofix-backup-<ts>`` copy
            before any fixer is allowed to write. Skipped when no fixer
            actually needs to write.
        apply: When False, run only the detection passes and report what
            would be fixed without mutating. Used by dry-run callers.

    Never raises. Fixer-specific failures are captured into the per-fix
    report so a single broken probe does not stop the others.
    """
    summary = AutoFixSummary(palace_path=palace_path)
    if not enabled:
        return summary

    db_path = os.path.join(palace_path, "chroma.sqlite3")
    if not os.path.isfile(db_path):
        return summary

    # Phase 1: detection — read-only, decides whether we need to mutate.
    pending: list[tuple[FixReport, callable]] = []
    sqlite_probes = (
        _probe_orphaned_queue_rows,
        _probe_orphaned_max_seq_id_rows,
        _probe_poisoned_max_seq_id,
    )
    for probe in sqlite_probes:
        report = FixReport(name=probe.__name__.replace("_probe_", ""))
        try:
            applier = probe(db_path, report)
        except Exception as exc:  # noqa: BLE001
            report.error = f"probe failed: {exc}"
            logger.debug("autofix probe %s failed", probe.__name__, exc_info=True)
            summary.reports.append(report)
            continue
        summary.reports.append(report)
        if report.detected and applier is not None:
            pending.append((report, applier))

    # Filesystem-level probe: payload-without-pickle partial HNSW segments
    # left by a process that exited before chromadb's sync_threshold flush.
    # These break the next add/upsert with "Error in compaction: Error
    # constructing hnsw segment reader". Surfacing through autofix means the
    # operator sees the quarantine action in status output instead of
    # discovering the silent rename only by ``ls``-ing the palace dir.
    fs_report = FixReport(name="partial_hnsw_segments")
    try:
        fs_applier = _probe_partial_hnsw_segments(palace_path, fs_report)
    except Exception as exc:  # noqa: BLE001
        fs_report.error = f"probe failed: {exc}"
        logger.debug("autofix probe partial_hnsw_segments failed", exc_info=True)
        summary.reports.append(fs_report)
    else:
        summary.reports.append(fs_report)
        if fs_report.detected and fs_applier is not None:
            pending.append((fs_report, fs_applier))

    # Phase 2: backup + apply, only if any probe found work AND apply is on.
    if not pending or not apply:
        return summary

    if backup:
        try:
            ts = time.strftime("%Y%m%d-%H%M%S")
            backup_path = os.path.join(palace_path, f"chroma.sqlite3.autofix-backup-{ts}")
            shutil.copy2(db_path, backup_path)
            summary.backup_path = backup_path
        except OSError as exc:
            # Backup failed — refuse to mutate.
            for report, _ in pending:
                report.error = f"backup failed before fix: {exc}"
            logger.warning("autofix backup failed: %s", exc)
            return summary

    for report, applier in pending:
        try:
            applier()
            report.applied = True
        except Exception as exc:  # noqa: BLE001
            report.error = f"apply failed: {exc}"
            logger.exception("autofix apply %s failed", report.name)

    return summary


# ---------------------------------------------------------------------------
# Fixers
# ---------------------------------------------------------------------------


def _probe_orphaned_queue_rows(db_path: str, report: FixReport):
    """Detect ``embeddings_queue`` rows whose topic references a removed collection.

    Chromadb routes writes by ``topic = persistent://default/default/<collection_uuid>``.
    When a collection is deleted (e.g., during ``mempalace repair``), the
    queue rows are not pruned and the compactor wastes work trying to process
    entries the segment reader cannot satisfy. Removing them is safe — the
    collection is gone, so any backfill against it would be a no-op anyway.
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            SELECT eq.topic, COUNT(*)
            FROM embeddings_queue eq
            WHERE NOT EXISTS (
                SELECT 1 FROM collections c
                WHERE eq.topic LIKE '%' || c.id || '%'
            )
            GROUP BY eq.topic
            """
        ).fetchall()
    if not rows:
        return None

    total = sum(c for _, c in rows)
    report.detected = True
    report.rows_affected = total
    report.detail = (
        f"{total} orphaned embeddings_queue rows across {len(rows)} dead "
        "collection topic(s)"
    )

    def _apply():
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                """
                DELETE FROM embeddings_queue
                WHERE NOT EXISTS (
                    SELECT 1 FROM collections c
                    WHERE embeddings_queue.topic LIKE '%' || c.id || '%'
                )
                """
            )
            report.rows_affected = cur.rowcount
            conn.commit()

    return _apply


def _probe_orphaned_max_seq_id_rows(db_path: str, report: FixReport):
    """Detect ``max_seq_id`` rows for segments that no longer exist.

    A stale row points at a deleted segment. Chromadb's compactor skips it
    on lookup but it clutters operator queries and the table is small
    enough that a targeted cleanup is essentially free.
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            """
            SELECT segment_id, seq_id
            FROM max_seq_id
            WHERE segment_id NOT IN (SELECT id FROM segments)
            """
        ).fetchall()
    if not rows:
        return None

    report.detected = True
    report.rows_affected = len(rows)
    report.detail = f"{len(rows)} max_seq_id row(s) pointing at deleted segments"

    def _apply():
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                """
                DELETE FROM max_seq_id
                WHERE segment_id NOT IN (SELECT id FROM segments)
                """
            )
            report.rows_affected = cur.rowcount
            conn.commit()

    return _apply


# Same threshold as the existing repair_max_seq_id command — anything above
# this is the 1.23e18 poisoning signature, not a real sequence id.
_POISONED_MAX_SEQ_ID_THRESHOLD = 1_000_000_000_000_000


def _probe_poisoned_max_seq_id(db_path: str, report: FixReport):
    """Detect ``max_seq_id`` rows poisoned by the 0.6.x BLOB→int misfire.

    See :func:`mempalace.repair.repair_max_seq_id` for the full history.
    Auto-fixes by recomputing each poisoned segment's seq_id from
    ``MAX(embeddings.seq_id)`` over its parent collection — exactly what
    the ``--mode max-seq-id`` command does, just inline and non-interactive.
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT segment_id, seq_id FROM max_seq_id WHERE seq_id > ?",
            (_POISONED_MAX_SEQ_ID_THRESHOLD,),
        ).fetchall()
    if not rows:
        return None

    report.detected = True
    report.rows_affected = len(rows)
    report.detail = (
        f"{len(rows)} poisoned max_seq_id row(s) (≥ 1.23e18); "
        "values will be recomputed from embeddings.seq_id"
    )

    def _apply():
        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            updated = 0
            for segment_id, _ in rows:
                fresh = _max_seq_id_for_segment(cur, segment_id)
                cur.execute(
                    "UPDATE max_seq_id SET seq_id = ? WHERE segment_id = ?",
                    (fresh, segment_id),
                )
                updated += cur.rowcount
            conn.commit()
            report.rows_affected = updated

    return _apply


def _max_seq_id_for_segment(cur: sqlite3.Cursor, segment_id: str) -> int:
    """``MAX(embeddings.seq_id)`` over the collection owning ``segment_id``.

    Mirrors :func:`mempalace.repair._compute_heuristic_seq_id` but inlined
    so the autofix module has no hard dependency on the repair module's
    private surface. Returns ``0`` when the segment is missing or the
    collection has no embeddings yet.
    """
    row = cur.execute(
        """
        SELECT MAX(e.seq_id)
        FROM embeddings e
        JOIN segments s ON e.segment_id = s.id
        WHERE s.collection = (
            SELECT collection FROM segments WHERE id = ?
        )
        """,
        (segment_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return 0
    val = row[0]
    if isinstance(val, (bytes, bytearray)):
        return int.from_bytes(val, "big")
    return int(val)


# ---------------------------------------------------------------------------
# Filesystem-level fixers
# ---------------------------------------------------------------------------


# Same floor used by ``backends.chroma.quarantine_partial_hnsw_segments``.
# Inlined here so the autofix module has no hard dependency on the backend's
# private constants — the values are unlikely to diverge because they both
# describe the same physical disk shape.
_AUTOFIX_PARTIAL_DATA_FLOOR = 1024


def _probe_partial_hnsw_segments(palace_path: str, report: FixReport):
    """Detect HNSW segment dirs left in a payload-without-pickle partial state.

    Mirrors the detection logic in :func:`backends.chroma.quarantine_partial_hnsw_segments`
    so the operator can see what would be cleaned without grepping the
    palace dir manually. Applier delegates to the backend function so there
    is one source of truth for the rename + logging.
    """
    try:
        entries = os.listdir(palace_path)
    except OSError:
        return None

    suspect_dirs: list[str] = []
    for name in entries:
        if (
            "-" not in name
            or name.startswith(".")
            or ".drift-" in name
            or ".corrupt-" in name
            or ".partial-" in name
        ):
            continue
        seg_dir = os.path.join(palace_path, name)
        if not os.path.isdir(seg_dir):
            continue

        data_path = os.path.join(seg_dir, "data_level0.bin")
        link_path = os.path.join(seg_dir, "link_lists.bin")
        meta_path = os.path.join(seg_dir, "index_metadata.pickle")

        try:
            data_size = os.path.getsize(data_path) if os.path.isfile(data_path) else 0
        except OSError:
            continue
        if data_size <= _AUTOFIX_PARTIAL_DATA_FLOOR:
            continue

        meta_ok = False
        if os.path.isfile(meta_path):
            try:
                meta_ok = os.path.getsize(meta_path) >= 16
            except OSError:
                meta_ok = False
        if meta_ok:
            continue

        link_size = 0
        if os.path.isfile(link_path):
            try:
                link_size = os.path.getsize(link_path)
            except OSError:
                link_size = 0
        if link_size > 0:
            continue

        suspect_dirs.append(seg_dir)

    if not suspect_dirs:
        return None

    report.detected = True
    report.rows_affected = len(suspect_dirs)
    short_list = ", ".join(os.path.basename(p) for p in suspect_dirs[:3])
    suffix = "" if len(suspect_dirs) <= 3 else f" (+{len(suspect_dirs) - 3} more)"
    report.detail = (
        f"{len(suspect_dirs)} partial HNSW segment dir(s) "
        f"(payload present, pickle missing/empty, link_lists empty): "
        f"{short_list}{suffix}"
    )

    def _apply():
        from .backends.chroma import quarantine_partial_hnsw_segments

        moved = quarantine_partial_hnsw_segments(palace_path)
        report.rows_affected = len(moved)

    return _apply


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_summary(summary: AutoFixSummary) -> list[str]:
    """Render an :class:`AutoFixSummary` as printable lines.

    Returns an empty list when nothing was detected, so callers can
    cheaply decide whether to print anything at all.
    """
    interesting = [r for r in summary.reports if r.detected or r.error]
    if not interesting:
        return []
    lines = ["  Auto-fix:"]
    for r in interesting:
        status = "applied" if r.applied else ("detected" if r.detected else "")
        if r.error:
            status = f"error — {r.error}"
        lines.append(f"    [{status}] {r.name}: {r.detail or '—'}")
    if summary.backup_path and summary.any_applied:
        lines.append(f"    backup: {summary.backup_path}")
    lines.append("")
    return lines
