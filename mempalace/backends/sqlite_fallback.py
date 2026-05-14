"""SQLite-only read fallback for diverged ChromaDB collections.

Background
----------
ChromaDB 1.5.x's Rust bindings segfault on ``count()``, ``get()``, and
``query()`` when the on-disk HNSW segment is critically diverged from
``chroma.sqlite3`` — the failure mode catalogued in #1222 and observed at
neo-cortex-mcp#2. The pickle metadata says the index holds N elements while
sqlite holds many multiples of N, so chromadb tries to load an HNSW that's
too small to map the queried IDs and the Rust binding crashes the entire
process instead of raising.

This module ships a read-only collection that bypasses chromadb entirely.
It reads documents and metadata directly from chroma.sqlite3 (which is
unaffected by HNSW corruption) so callers like ``wake_up`` continue to
function while the operator runs ``mempalace repair --mode rebuild``.

The fallback supports the operator subset actually used by mempalace
internals (verified against ``searcher.build_where_filter`` and the layer
callers): bare ``{key: value}`` equality, ``$and``, ``$or``, ``$eq``,
``$ne``, ``$in``, ``$nin``, ``$gt``/``$gte``/``$lt``/``$lte``, plus
``$contains`` for ``where_document``. Anything else raises
``UnsupportedFilterError`` rather than silently dropping the predicate.

Writes (``add``, ``upsert``, ``update``, ``delete``) and semantic search
(``query``) raise :class:`CollectionDegradedError` so callers fail fast
with a clear repair hint. Allowing writes against a corrupt index would
push more rows into a collection that vector search cannot reach and
deepen the divergence.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Optional

from .base import (
    BackendError,
    BaseCollection,
    GetResult,
    HealthStatus,
    UnsupportedFilterError,
    _IncludeSpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class CollectionDegradedError(BackendError):
    """Raised when a collection's HNSW is too divergent for chromadb to serve.

    Reads via the sqlite fallback still work; writes and semantic search do
    not. The error message names the operation and tells the operator how
    to recover.
    """


# ---------------------------------------------------------------------------
# Divergence cache
# ---------------------------------------------------------------------------


# ``(palace_path, collection_name) -> bool`` — True when the collection is
# in critical-divergence territory and must be served from sqlite. Cached
# per process because the probe reads both ``chroma.sqlite3`` and the HNSW
# metadata pickle, and we don't want every ``get_collection`` call to pay
# that cost. Cleared on explicit ``invalidate_divergence_cache`` (after a
# repair) and on backend ``close``.
_DIVERGENCE_CACHE: dict[tuple[str, str], bool] = {}
_DIVERGENCE_LOCK = threading.Lock()


def is_critically_diverged(palace_path: str, collection_name: str) -> Optional[dict]:
    """Return the divergence status dict when the collection is unsafe to open via chromadb.

    Returns ``None`` when the collection looks healthy. The non-None return
    is the full ``hnsw_capacity_status`` result so callers can surface the
    operator-facing message verbatim.

    Cached per ``(palace_path, collection_name)`` per process. To force a
    re-probe (e.g., after running repair), call
    :func:`invalidate_divergence_cache`.
    """
    key = (palace_path, collection_name)
    with _DIVERGENCE_LOCK:
        cached = _DIVERGENCE_CACHE.get(key)
        if cached is False:
            return None
        if cached is True:
            # Re-fetch the message so it stays informative across calls.
            from .chroma import hnsw_capacity_status

            return hnsw_capacity_status(palace_path, collection_name)

    # First probe: compute outside the lock so two callers don't serialize
    # on a slow disk read. The map is idempotent — duplicate writes are fine.
    from .chroma import hnsw_capacity_status

    status = hnsw_capacity_status(palace_path, collection_name)
    diverged = bool(status.get("diverged"))
    with _DIVERGENCE_LOCK:
        _DIVERGENCE_CACHE[key] = diverged
    return status if diverged else None


def invalidate_divergence_cache(palace_path: Optional[str] = None) -> None:
    """Clear cached divergence verdicts.

    ``palace_path=None`` clears every entry — used by tests and on backend
    ``close``. Passing a path clears only entries for that palace, used by
    ``mempalace repair`` after a successful rebuild.
    """
    with _DIVERGENCE_LOCK:
        if palace_path is None:
            _DIVERGENCE_CACHE.clear()
            return
        for k in list(_DIVERGENCE_CACHE.keys()):
            if k[0] == palace_path:
                _DIVERGENCE_CACHE.pop(k, None)


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


class SqliteReadOnlyCollection(BaseCollection):
    """Read-only ``BaseCollection`` backed by chroma.sqlite3 directly.

    Exists to keep wake_up / status / non-vector lookups working when the
    HNSW segment is too divergent for chromadb to load without segfaulting.
    Every chromadb call is replaced with a stdlib ``sqlite3`` read.
    """

    def __init__(self, palace_path: str, collection_name: str, status_message: str):
        self._palace_path = palace_path
        self._collection_name = collection_name
        self._db_path = os.path.join(palace_path, "chroma.sqlite3")
        self._status_message = status_message or "HNSW index divergent"

    # ------------------------------------------------------------------
    # Refusals
    # ------------------------------------------------------------------

    def _degraded(self, op: str) -> CollectionDegradedError:
        return CollectionDegradedError(
            f"{op} unavailable for '{self._collection_name}': {self._status_message}. "
            "Run `mempalace repair --mode rebuild` to restore full operation."
        )

    def add(self, **_):
        raise self._degraded("add")

    def upsert(self, **_):
        raise self._degraded("upsert")

    def update(self, **_):
        raise self._degraded("update")

    def delete(self, **_):
        raise self._degraded("delete")

    def query(self, **_):
        raise self._degraded("query (semantic search)")

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def count(self) -> int:
        return _sqlite_drawer_count(self._db_path, self._collection_name) or 0

    def estimated_count(self) -> int:
        return self.count()

    def get(
        self,
        *,
        ids: Optional[list[str]] = None,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        include: Optional[list[str]] = None,
    ) -> GetResult:
        return _sqlite_get(
            self._db_path,
            self._collection_name,
            ids=ids,
            where=where,
            where_document=where_document,
            limit=limit,
            offset=offset,
            include=include,
        )

    def health(self) -> HealthStatus:
        return HealthStatus.unhealthy(self._status_message)

    # Compat: ChromaCollection exposes ``.metadata`` (searcher uses it to
    # detect legacy non-cosine palaces).
    @property
    def metadata(self) -> dict:
        return _sqlite_collection_metadata(self._db_path, self._collection_name)


# ---------------------------------------------------------------------------
# SQLite query helpers
# ---------------------------------------------------------------------------


def _sqlite_drawer_count(db_path: str, collection_name: str) -> Optional[int]:
    if not os.path.isfile(db_path):
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM embeddings e
                JOIN segments s ON e.segment_id = s.id
                JOIN collections c ON s.collection = c.id
                WHERE c.name = ? AND s.scope = 'METADATA'
                """,
                (collection_name,),
            ).fetchone()
            return int(row[0]) if row and row[0] is not None else None
        finally:
            conn.close()
    except sqlite3.Error:
        logger.debug("_sqlite_drawer_count failed for %s", collection_name, exc_info=True)
        return None


def _sqlite_collection_metadata(db_path: str, collection_name: str) -> dict:
    if not os.path.isfile(db_path):
        return {}
    out: dict = {}
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            cur = conn.execute(
                """
                SELECT cm.key, cm.str_value, cm.int_value, cm.float_value, cm.bool_value
                FROM collection_metadata cm
                JOIN collections c ON cm.collection_id = c.id
                WHERE c.name = ?
                """,
                (collection_name,),
            )
            rows = cur.fetchall()
        finally:
            conn.close()
    except sqlite3.OperationalError:
        # ChromaDB renamed columns in some versions (string_value vs str_value);
        # try the alternate spelling before giving up.
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            try:
                cur = conn.execute(
                    """
                    SELECT cm.key, cm.string_value, cm.int_value, cm.float_value, cm.bool_value
                    FROM collection_metadata cm
                    JOIN collections c ON cm.collection_id = c.id
                    WHERE c.name = ?
                    """,
                    (collection_name,),
                )
                rows = cur.fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            return {}
    except sqlite3.Error:
        return {}

    for key, s, i, f, b in rows:
        if s is not None:
            out[key] = s
        elif i is not None:
            out[key] = i
        elif f is not None:
            out[key] = f
        elif b is not None:
            out[key] = bool(b)
    return out


def _sqlite_get(
    db_path: str,
    collection_name: str,
    *,
    ids,
    where,
    where_document,
    limit,
    offset,
    include,
) -> GetResult:
    """SQLite implementation of ``ChromaCollection.get``."""
    spec = _IncludeSpec.resolve(include, default_distances=False)

    if spec.embeddings:
        # The HNSW divergence is exactly the reason this fallback exists —
        # the vector data is unreliable. Refuse rather than return garbage.
        raise CollectionDegradedError(
            "embeddings not available in sqlite fallback for "
            f"'{collection_name}': HNSW index is divergent. "
            "Run `mempalace repair --mode rebuild`."
        )

    if not os.path.isfile(db_path):
        return GetResult.empty()

    where_sql, where_params = _where_to_sql(where) if where else ("", [])
    doc_sql, doc_params = _where_document_to_sql(where_document) if where_document else ("", [])

    base = [
        "SELECT e.id, e.embedding_id",
        "FROM embeddings e",
        "JOIN segments s ON e.segment_id = s.id",
        "JOIN collections c ON s.collection = c.id",
        "WHERE c.name = ? AND s.scope = 'METADATA'",
    ]
    params: list = [collection_name]

    if ids is not None:
        if not ids:
            return GetResult.empty()
        placeholders = ",".join("?" * len(ids))
        base.append(f"AND e.embedding_id IN ({placeholders})")
        params.extend(ids)

    if where_sql:
        base.append(f"AND {where_sql}")
        params.extend(where_params)

    if doc_sql:
        base.append(f"AND e.id IN ({doc_sql})")
        params.extend(doc_params)

    base.append("ORDER BY e.id")
    if limit is not None:
        base.append("LIMIT ?")
        params.append(int(limit))
        if offset is not None:
            base.append("OFFSET ?")
            params.append(int(offset))

    sql = " ".join(base)

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        logger.warning("sqlite fallback: cannot open %s: %s", db_path, exc)
        return GetResult.empty()

    try:
        try:
            cur = conn.execute(sql, params)
            rows = cur.fetchall()
        except sqlite3.Error as exc:
            logger.warning("sqlite fallback: query failed: %s\nSQL: %s", exc, sql)
            return GetResult.empty()

        if not rows:
            return GetResult.empty()

        e_ids = [r[0] for r in rows]
        out_ids = [r[1] for r in rows]
        idx_by_eid = {eid: i for i, eid in enumerate(e_ids)}

        out_docs = [""] * len(out_ids) if spec.documents else []
        out_metas: list[dict] = [dict() for _ in out_ids] if spec.metadatas else []

        if spec.documents or spec.metadatas:
            placeholders = ",".join("?" * len(e_ids))
            meta_cur = conn.execute(
                f"""
                SELECT id, key, string_value, int_value, float_value, bool_value
                FROM embedding_metadata
                WHERE id IN ({placeholders})
                """,
                e_ids,
            )
            for eid, key, sv, iv, fv, bv in meta_cur.fetchall():
                pos = idx_by_eid.get(eid)
                if pos is None:
                    continue
                if key == "chroma:document":
                    if spec.documents:
                        out_docs[pos] = sv or ""
                    continue
                if not spec.metadatas:
                    continue
                if sv is not None:
                    out_metas[pos][key] = sv
                elif iv is not None:
                    out_metas[pos][key] = iv
                elif fv is not None:
                    out_metas[pos][key] = fv
                elif bv is not None:
                    out_metas[pos][key] = bool(bv)
    finally:
        conn.close()

    return GetResult(
        ids=out_ids,
        documents=out_docs,
        metadatas=out_metas,
        embeddings=None,
    )


# ---------------------------------------------------------------------------
# where clause -> SQL
# ---------------------------------------------------------------------------


def _where_to_sql(where: dict) -> tuple[str, list]:
    """Render a chromadb-style where clause as a SQL boolean predicate on ``e``.

    The result is a fragment intended for use as ``AND (<fragment>)`` in a
    query over ``embeddings e``. Always emits ``EXISTS`` subqueries against
    ``embedding_metadata`` keyed by ``e.id`` so the metadata join stays
    correctly scoped per drawer.
    """
    if not where:
        return "", []

    keys = list(where.keys())
    op_keys = [k for k in keys if k.startswith("$")]
    if op_keys:
        if len(keys) != 1:
            raise UnsupportedFilterError(
                f"compound where clause cannot mix operators and bare keys: {keys}"
            )
        op = op_keys[0]
        val = where[op]
        if op == "$and":
            return _join_sub_predicates(val, " AND ")
        if op == "$or":
            return _join_sub_predicates(val, " OR ")
        raise UnsupportedFilterError(f"top-level operator {op!r} not supported")

    # Bare keys: implicit $and.
    parts: list[str] = []
    params: list = []
    for key, value in where.items():
        sub_sql, sub_params = _key_predicate_sql(key, value)
        parts.append(f"({sub_sql})")
        params.extend(sub_params)
    return " AND ".join(parts), params


def _join_sub_predicates(subs, joiner: str) -> tuple[str, list]:
    if not isinstance(subs, list):
        raise UnsupportedFilterError("$and / $or value must be a list")
    parts: list[str] = []
    params: list = []
    for sub in subs:
        if not isinstance(sub, dict):
            raise UnsupportedFilterError("$and / $or entries must be dicts")
        sub_sql, sub_params = _where_to_sql(sub)
        if not sub_sql:
            continue
        parts.append(f"({sub_sql})")
        params.extend(sub_params)
    if not parts:
        return "", []
    return joiner.join(parts), params


def _key_predicate_sql(key: str, value) -> tuple[str, list]:
    """Translate a single ``{key: value-or-op-dict}`` to SQL."""
    if isinstance(value, dict):
        if len(value) != 1:
            raise UnsupportedFilterError(f"value for {key!r} must have exactly one operator")
        op, op_val = next(iter(value.items()))
        return _op_to_exists(key, op, op_val)
    # Bare scalar -> $eq.
    return _op_to_exists(key, "$eq", value)


def _op_to_exists(key: str, op: str, val) -> tuple[str, list]:
    """Render a single (key, op, value) predicate as ``EXISTS (...)``."""
    base = (
        "EXISTS (SELECT 1 FROM embedding_metadata m "
        "WHERE m.id = e.id AND m.key = ?"
    )

    if op in ("$eq", "$ne"):
        col, py_val = _value_column(val)
        cmp = "=" if op == "$eq" else "<>"
        return base + f" AND m.{col} {cmp} ?)", [key, py_val]

    if op in ("$gt", "$gte", "$lt", "$lte"):
        col, py_val = _value_column(val)
        sym = {"$gt": ">", "$gte": ">=", "$lt": "<", "$lte": "<="}[op]
        return base + f" AND m.{col} {sym} ?)", [key, py_val]

    if op in ("$in", "$nin"):
        if not isinstance(val, list):
            raise UnsupportedFilterError(f"{op} value for {key!r} must be a list")
        if not val:
            # Chroma's behaviour: $in [] matches nothing, $nin [] matches all.
            if op == "$in":
                return "0 = 1", []
            return base + ")", [key]
        cols = {_value_column(v)[0] for v in val}
        if len(cols) != 1:
            raise UnsupportedFilterError(f"{op} values for {key!r} must share one type")
        col = next(iter(cols))
        placeholders = ",".join("?" * len(val))
        cmp = "IN" if op == "$in" else "NOT IN"
        py_vals = [_value_column(v)[1] for v in val]
        return base + f" AND m.{col} {cmp} ({placeholders}))", [key] + py_vals

    if op == "$contains":
        if not isinstance(val, str):
            raise UnsupportedFilterError("$contains operand must be a string")
        like = _like_pattern(val)
        return (
            base + " AND m.string_value LIKE ? ESCAPE '\\')",
            [key, like],
        )

    raise UnsupportedFilterError(f"operator {op!r} not supported in sqlite fallback")


def _where_document_to_sql(where_document: dict) -> tuple[str, list]:
    """Translate a where_document clause to a sub-SELECT yielding embedding ids."""
    if not where_document:
        return "", []
    if len(where_document) != 1:
        raise UnsupportedFilterError("where_document must have exactly one operator")
    op = next(iter(where_document))
    val = where_document[op]
    if op == "$contains":
        if not isinstance(val, str):
            raise UnsupportedFilterError("$contains operand must be a string")
        like = _like_pattern(val)
        return (
            "SELECT id FROM embedding_metadata "
            "WHERE key = 'chroma:document' AND string_value LIKE ? ESCAPE '\\'",
            [like],
        )
    raise UnsupportedFilterError(f"where_document operator {op!r} not supported")


def _value_column(v):
    if isinstance(v, bool):
        return "bool_value", 1 if v else 0
    if isinstance(v, int):
        return "int_value", v
    if isinstance(v, float):
        return "float_value", v
    return "string_value", str(v)


def _like_pattern(needle: str) -> str:
    # Escape backslash first so it doesn't double-process the substitutions below.
    escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"
