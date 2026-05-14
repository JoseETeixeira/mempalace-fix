# mempalace-fix

A maintained fork of [`mempalace`](https://github.com/milla-jovovich/mempalace)
that fixes the SIGSEGV on `wake-up` (and every other read path) when the
ChromaDB HNSW index is critically diverged from the SQLite source of truth,
plus the related partial-write segment failure mode.

It is layered on top of upstream `develop` (version `3.3.5`) so it already
carries the `chromadb` metadata sanitizer, the cold-start HNSW pickle
quarantine, the SQLite integrity preflight, the per-target hook PID guard,
the Windows popen detach fix, and the rest of the upstream `develop` fixes.

## Install

```bash
pip install --user git+https://github.com/JoseETeixeira/mempalace-fix.git
```

If you have the PyPI build installed, swap it over with:

```bash
pip uninstall mempalace
pip install --user git+https://github.com/JoseETeixeira/mempalace-fix.git
```

The Python package keeps the upstream name (`mempalace`), so `pip show
mempalace` and `python3 -m mempalace ...` continue to work unchanged.

`coding-cli setup full` performs the swap automatically on every run.

## What this fork adds on top of upstream

| Area | Change |
|------|--------|
| Read path | Diverged collections fall back to a `SqliteReadOnlyCollection` that reads `chroma.sqlite3` directly, bypassing ChromaDB's Rust HNSW loader so `count`/`get`/`query` no longer segfault. |
| MCP server | `_get_collection` honours the existing `_vector_disabled` flag for every tool, not just `search` — previously the other tools would still crash on the next `col.count()`. |
| Cold start | `quarantine_partial_hnsw_segments` detects payload-without-pickle segment dirs (the partial-write shape ChromaDB 1.5.x leaves behind when a process exits before `hnsw:sync_threshold`) and moves them aside before any open. |
| Runtime | `ChromaCollection` promotes itself to the SQLite fallback in place when ChromaDB raises an `InternalError` matching `_HNSW_RUNTIME_LOAD_MARKERS`, so a freshly rebuilt index that the reader still refuses to map doesn't take the process down. |
| Auto-fix | New `mempalace/autofix.py` runs every safe, idempotent fixer (`orphaned_queue_rows`, `orphaned_max_seq_id_rows`, `poisoned_max_seq_id`, `partial_hnsw_segments`) on every `mempalace status` and reports what it touched. Backs up `chroma.sqlite3` before mutating. |
| Search UX | `mempalace search ...` and the MCP `search_memories` tool auto-route to BM25 over SQLite when the vector path is unavailable, instead of returning an error and a repair hint. Vector search resumes automatically once the segment becomes loadable again. |

See [`CHANGES.md`](./CHANGES.md) for the full failure analysis, stack
traces, and per-file diff summary.

## Upstream

Upstream lives at <https://github.com/milla-jovovich/mempalace>. This fork
tracks `develop` (currently 3.3.5) and is intended to fold back when the
upstream maintainers pick up the SQLite fallback + auto-fix patches. Open
an issue here if you spot drift.

## Layout

```
mempalace-fix/
  mempalace/        Python package — see mempalace/README.md for a module map.
  CHANGES.md        Detailed change log for this fork.
  README.md         (this file)
```
