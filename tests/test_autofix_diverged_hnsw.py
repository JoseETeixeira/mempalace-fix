"""Smoke tests for the silent-HNSW-divergence autofix probe.

Covers the on-disk shape that ``mempalace repair --mode hnsw-auto`` used
to ignore (sqlite has many embeddings, HNSW pickle reports few) — the
operator-visible failure described in CHANGES.md § "autofix also detects
silent HNSW index divergence".

Stubs are used heavily so the tests stay hermetic: real
``hnsw_capacity_status`` would need a chromadb client + on-disk segment
dirs, and ``rebuild_from_sqlite`` would re-embed every drawer. Both
would dwarf the test's CI budget. The probe under test is pure logic
over the dict returned by ``hnsw_capacity_status``, so monkey-patching
that helper is enough to exercise every branch.
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
import unittest
from typing import Any


# Make ``mempalace`` importable when the tests are run as
# ``python -m unittest discover -s tests`` from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


from mempalace import autofix  # noqa: E402


def _make_palace(tmp_path: str) -> str:
    """Create a minimal palace dir with an empty ``chroma.sqlite3``.

    ``auto_fix_palace`` short-circuits when the sqlite file is missing,
    so the file must exist even if it has no tables. The sqlite probes
    are also stubbed in these tests, so an empty file is safe.
    """
    db_path = os.path.join(tmp_path, "chroma.sqlite3")
    conn = sqlite3.connect(db_path)
    try:
        # An empty file passes ``os.path.isfile`` but ``sqlite3.connect``
        # creates a real header so future calls don't complain.
        conn.execute("PRAGMA user_version = 0")
        conn.commit()
    finally:
        conn.close()
    return tmp_path


class _CapacityStub:
    """Tracks which collection names were queried + returns canned responses."""

    def __init__(self, by_collection: dict[str, dict[str, Any]]):
        self.by_collection = by_collection
        self.calls: list[str] = []

    def __call__(self, palace_path: str, collection_name: str) -> dict[str, Any]:
        self.calls.append(collection_name)
        return self.by_collection.get(
            collection_name,
            {
                "segment_id": "stub-seg",
                "sqlite_count": 100,
                "hnsw_count": 100,
                "divergence": 0,
                "diverged": False,
                "status": "ok",
                "message": "stub-ok",
            },
        )


class DivergedHnswProbeTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.palace_path = _make_palace(self._tmp.name)

        # Stub the three sqlite probes so they always report a clean
        # palace — this test isolates the diverged-index probe.
        self._original_sqlite_probes = autofix.auto_fix_palace.__globals__[
            "_probe_orphaned_queue_rows"
        ]
        self._patch_sqlite_probes()

        # Stub the partial-segment probe (filesystem walk) the same way.
        self._original_partial = autofix._probe_partial_hnsw_segments
        autofix._probe_partial_hnsw_segments = lambda *a, **k: None
        self.addCleanup(
            setattr, autofix, "_probe_partial_hnsw_segments", self._original_partial
        )

        # Stub the stuck-writer probe so the new probe is the only
        # heavy detector in play.
        self._original_writer = autofix._probe_degraded_hnsw_writer
        autofix._probe_degraded_hnsw_writer = lambda *a, **k: None
        self.addCleanup(
            setattr,
            autofix,
            "_probe_degraded_hnsw_writer",
            self._original_writer,
        )

    def _patch_sqlite_probes(self):
        for name in (
            "_probe_orphaned_queue_rows",
            "_probe_orphaned_max_seq_id_rows",
            "_probe_poisoned_max_seq_id",
        ):
            original = getattr(autofix, name)
            setattr(autofix, name, lambda *a, **k: None)
            self.addCleanup(setattr, autofix, name, original)

    def _install_capacity_stub(self, stub: _CapacityStub):
        """Inject ``_CapacityStub`` into the autofix module's import path.

        The probe imports ``hnsw_capacity_status`` lazily inside its
        function body via ``from .backends.chroma import ...``. We
        replace the symbol on the actual module so the import resolves
        to our stub.
        """
        import mempalace.backends.chroma as chroma_module

        original = chroma_module.hnsw_capacity_status
        chroma_module.hnsw_capacity_status = stub
        self.addCleanup(
            setattr, chroma_module, "hnsw_capacity_status", original
        )

    def test_healthy_palace_reports_no_detection(self):
        stub = _CapacityStub({})  # default branch returns ok
        self._install_capacity_stub(stub)

        summary = autofix.auto_fix_palace(self.palace_path, allow_heavy=True)

        report = next(
            r for r in summary.reports if r.name == "diverged_hnsw_index"
        )
        self.assertFalse(report.detected)
        self.assertIsNone(report.error)
        # Both collections must be queried — closets divergence is just
        # as service-impacting as drawers divergence.
        self.assertEqual(len(stub.calls), 2)

    def test_diverged_drawers_detected_and_gated(self):
        stub = _CapacityStub(
            {
                "mempalace_drawers": {
                    "segment_id": "seg-d",
                    "sqlite_count": 274_199,
                    "hnsw_count": 27_121,
                    "divergence": 247_078,
                    "diverged": True,
                    "status": "diverged",
                    "message": "...",
                },
            }
        )
        self._install_capacity_stub(stub)

        # allow_heavy=False (the default for ``status`` / ``miner``)
        summary = autofix.auto_fix_palace(self.palace_path, allow_heavy=False)
        report = next(
            r for r in summary.reports if r.name == "diverged_hnsw_index"
        )
        self.assertTrue(report.detected)
        self.assertFalse(report.applied)
        self.assertIn("mempalace_drawers", report.detail)
        self.assertIn("247,078", report.detail)
        # The hint must be appended so operators discover the recovery
        # command from the same line that reports the divergence.
        self.assertIn("mempalace repair --mode hnsw-auto", report.detail)
        # rows_affected uses the worst gap across collections.
        self.assertEqual(report.rows_affected, 247_078)

    def test_diverged_closets_detected(self):
        stub = _CapacityStub(
            {
                "mempalace_closets": {
                    "segment_id": "seg-c",
                    "sqlite_count": 5_729,
                    "hnsw_count": 0,
                    "divergence": 5_729,
                    "diverged": True,
                    "status": "diverged",
                    "message": "...",
                },
            }
        )
        self._install_capacity_stub(stub)

        summary = autofix.auto_fix_palace(self.palace_path, allow_heavy=True)
        report = next(
            r for r in summary.reports if r.name == "diverged_hnsw_index"
        )
        self.assertTrue(report.detected)
        self.assertIn("mempalace_closets", report.detail)

    def test_capacity_probe_exception_does_not_break_autofix(self):
        # If hnsw_capacity_status raises (e.g. transient sqlite lock),
        # the new probe must still return a clean "no detection" verdict
        # rather than blowing up the entire autofix pass.
        import mempalace.backends.chroma as chroma_module

        def _boom(palace_path, collection_name):
            raise RuntimeError("sqlite locked")

        original = chroma_module.hnsw_capacity_status
        chroma_module.hnsw_capacity_status = _boom
        self.addCleanup(
            setattr, chroma_module, "hnsw_capacity_status", original
        )

        summary = autofix.auto_fix_palace(self.palace_path, allow_heavy=True)
        report = next(
            r for r in summary.reports if r.name == "diverged_hnsw_index"
        )
        self.assertFalse(report.detected)
        # Per-collection exceptions are swallowed inside the probe (see
        # the ``logger.debug`` branch), so report.error stays empty.
        self.assertIsNone(report.error)


if __name__ == "__main__":
    unittest.main()
