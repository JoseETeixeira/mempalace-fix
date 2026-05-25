# mempalace fix: develop branch (3.3.5) + sqlite fallback + auto-fix on top

## Base

This patchset is layered on top of the upstream `develop` branch (version
`3.3.5`), which already includes critical upstream fixes that are highly
relevant to the failure mode we hit:

- **`5d95656` + `a26e465`** — chromadb upsert metadata sanitizer (PR #1458).
  Coerces `None` / `{}` metadata entries to `{"_repaired_empty_meta": True}`
  before they reach chromadb. Prevents the `ValueError: Expected metadata to
  be a non-empty dict` crash at ~120K drawers during rebuild.
- **`quarantine_invalid_hnsw_metadata`** (in `backends/chroma.py`) — eagerly
  detects `index_metadata.pickle` files with corrupted fields (e.g.
  `dimensionality=None`, the exact signature we hit after our first
  rebuild) and renames them aside before chromadb tries to load. Chroma
  opens against an empty index and lazily backfills from the embeddings
  queue instead of segfaulting.
- **`5134a63`** — SQLite integrity preflight before chromadb open.
- **`251c5a0`** — `rebuild_index` accepts a progress callback (default
  prints an ETA), much better UX for the ~30-60 min rebuild on large
  palaces.
- **`3a76360`** — per-target PID guard with atomic claim for hooks
  (concurrency bug fix).
- **`bc7392a`** — Windows popen detach fix.
- **`74cd1c3`** — `convo_miner` bulk pre-fetch of already-mined set
  (replaces N WHERE queries).
- **`1d3eecb`** — gitignore-aware drawer prune (new `sync.py` module).
- **`e334e25` / `be95ea7`** — retry transient chromadb errors in
  `_get_collection` and `tool_search`.
- **`733e435`** — guard `searcher` against `None` metadata/doc.
- **`f2bed92`** — clamp layers similarity to `[0, 1]`.
- **`f28d0e3` + `982adcf`** — structured MCP error for param-shape mismatch.

# mempalace fix: SIGSEGV on wake-up when HNSW index is critically diverged

## Failure observed

```
$ mempalace wake-up
zsh: segmentation fault  python3 -m mempalace wake-up
```

Stack trace via `python3 -X faulthandler -m mempalace wake-up`:

```
File "chromadb/api/rust.py", line 440 in _get
File "chromadb/api/models/Collection.py", line 161 in get
File "mempalace/backends/chroma.py", line 840 in get
File "mempalace/layers.py", line 108 in generate
File "mempalace/layers.py", line 394 in wake_up
File "mempalace/cli.py", line 589 in cmd_wakeup
```

Even `chromadb.Collection.count()` segfaults — not just `.get()` or `.query()`.

## Root cause

The `mempalace_drawers` ChromaDB collection on the affected palace
(`/Users/edu/.mempalace/palace`) has a critically divergent HNSW segment:

- `chroma.sqlite3.embeddings` rows for the collection: **145,260**
- HNSW `index_metadata.pickle` `id_to_label` entries:           **959**
- Missing in the index:        **144,301 (99%)**

This is the same failure mode catalogued in mempalace #1222 (16,384 of
192,997) and neo-cortex-mcp #2 (SIGSEGV on `count()` with chromadb 1.5.5).
ChromaDB 1.5.x's Rust bindings load HNSW eagerly on any collection op —
including `count()` — and crash the entire process when the on-disk index
cannot map the rows sqlite says exist.

`hnsw_capacity_status()` already detects this and `mempalace repair-status`
already prints a clear DIVERGED verdict. The MCP server already gates the
`search` tool on a `_vector_disabled` flag. But `wake-up`, `status` (CLI),
`recall`, and the MCP server's read tools all call `col.count()` /
`col.get()` directly on the chromadb collection, so the divergence detection
existed but did not actually prevent the segfault.

## Fix

Bypass ChromaDB's Rust path entirely for diverged collections. Reads now go
through stdlib `sqlite3` against `chroma.sqlite3`, which is the source of
truth chromadb itself uses for `embeddings` + `embedding_metadata`. Writes
and semantic search refuse with a clear repair hint.

### Files

- **NEW** `mempalace/backends/sqlite_fallback.py`
  - `CollectionDegradedError` — raised by writes/query in fallback mode
  - `SqliteReadOnlyCollection` — implements `BaseCollection`, reads from sqlite
  - `is_critically_diverged(palace_path, collection_name)` — cached probe
  - `invalidate_divergence_cache(palace_path=None)` — clear after repair
  - SQL translator for the chromadb `where` subset mempalace actually uses
    (`$eq`, `$ne`, `$in`, `$nin`, `$and`, `$or`, `$gt/$gte/$lt/$lte`,
    `$contains`, plus bare `{key: value}`)

- **MODIFIED** `mempalace/backends/chroma.py`
  - `ChromaBackend.get_collection`: pre-flight divergence check; returns
    `SqliteReadOnlyCollection` when diverged so chromadb is never asked to
    open the broken segment. Passes palace context to `ChromaCollection`
    so runtime fallback can promote in place.
  - `ChromaBackend.delete_collection`: invalidates divergence cache so a
    fresh post-repair collection is treated as healthy without restart.
  - `ChromaBackend.close`: clears divergence cache.
  - `ChromaCollection`: new **runtime fallback** for `count` / `get` /
    `query`. When chromadb raises an `InternalError` matching
    `_HNSW_RUNTIME_LOAD_MARKERS` (e.g., "Error loading hnsw index" — fires
    when chromadb's Rust compactor can't open a segment even though the
    pre-flight pickle check passes), the collection promotes itself to the
    sqlite fallback in place. Subsequent reads on the same instance skip
    chromadb entirely; the per-process divergence cache is also set so the
    next `get_collection` short-circuits to sqlite without re-paying the
    chromadb failure. Catches the failure mode where a fresh `repair
    --mode legacy` rebuild produces an HNSW that chromadb's reader
    refuses to map.

- **MODIFIED** `mempalace/mcp_server.py`
  - `_get_collection`: when `_vector_disabled` is set (the existing
    capacity-probe flag), return a `SqliteReadOnlyCollection` instead of a
    ChromaCollection. Previously the flag only gated the `search` tool;
    every other MCP tool segfaulted on the next `col.count()`.

- **NEW** `quarantine_partial_hnsw_segments` in `mempalace/backends/chroma.py`
  - Detects segment dirs left in the payload-without-pickle partial-write
    shape (`data_level0.bin > 1 KiB`, `index_metadata.pickle` missing or
    < 16 bytes, `link_lists.bin` empty). Chromadb 1.5.x buffers up to
    `hnsw:sync_threshold` (50 000 in our config) additions before flushing
    the pickle, so any process exit before that threshold leaves a segment
    on disk that the Rust compactor cannot reopen. The next `add`/`upsert`
    raises `chromadb.errors.InternalError: Error in compaction: Error
    constructing hnsw segment reader` — the literal "save FAIL — HNSW
    index broken" failure mode observed during the patch session.
  - Wired into `ChromaBackend._prepare_palace_for_open` alongside the
    existing `quarantine_invalid_hnsw_metadata` and `quarantine_stale_hnsw`
    cold-start probes. Every cold-start opens against either a healthy
    segment or a clean empty one — never against a half-written one.
  - Also exposed via the autofix module so `mempalace status` reports what
    was quarantined instead of doing it silently.

- **NEW** `mempalace/autofix.py`
  - `auto_fix_palace(palace_path)` — runs every safe, idempotent fixer and
    returns an `AutoFixSummary`. Never raises; per-fix errors are captured
    into the report. Mutates only when a probe finds work, and always
    takes a `chroma.sqlite3.autofix-backup-<ts>` copy first so the apply
    pass is reversible.
  - Fixers shipped:
    1. `orphaned_queue_rows` — deletes `embeddings_queue` rows whose
       topic references a collection no longer in the `collections`
       table (post-rebuild residue that wastes compactor work).
    2. `orphaned_max_seq_id_rows` — deletes `max_seq_id` rows pointing
       at segments that no longer exist.
    3. `poisoned_max_seq_id` — detects rows ≥ 1.23e18 (the 0.6.x BLOB→int
       shim signature) and recomputes them from `MAX(embeddings.seq_id)`
       over the parent collection. Same logic as
       `mempalace repair --mode max-seq-id`, applied automatically.
    4. `partial_hnsw_segments` — quarantines HNSW segment dirs left in the
       payload-without-pickle partial-write shape (see the
       `quarantine_partial_hnsw_segments` section above). Mirrors the same
       detection as the cold-start pre-open probe so the operator sees the
       quarantine action in `mempalace status` output instead of having to
       discover it by listing the palace dir.
  - `render_summary(...)` — returns printable lines for callers (empty
    list when nothing detected, so callers can opt into output cheaply).

- **MODIFIED** `mempalace/miner.py`
  - `status(palace_path)`: invokes `auto_fix_palace` before the count.
    When fixers found nothing, output is unchanged. When fixers applied
    something, the report appears above the count block and the backup
    path is shown for auditing.

- **MODIFIED** `mempalace/searcher.py`
  - `search` (CLI path): on `CollectionDegradedError` or when the backend
    returned a `SqliteReadOnlyCollection` directly, auto-routes to the
    existing `_bm25_only_via_sqlite` path and prints results in the same
    layout as the vector path (with a "(BM25 sqlite fallback)" header so
    users can tell). No more repair hint at the search prompt.
  - `search_memories` (MCP path): on `CollectionDegradedError` or a
    runtime HNSW-load error string from chromadb's Rust binding, auto-
    routes to `_bm25_only_via_sqlite`. Result dict gains
    `auto_fallback: "bm25_sqlite"` so callers can tell the search was
    served from sqlite without the operator having to flip a flag.
  - New `_print_bm25_results` helper renders the BM25-sqlite dict in the
    CLI search layout.

### Behavior after fix

- `mempalace wake-up` → reads documents via sqlite, returns L0+L1 text.
- `mempalace status` → counts and groups via sqlite.
- `mempalace search "..."` → returns BM25-sqlite results automatically
  when vector search is unavailable. The user gets a results list (with a
  "(BM25 sqlite fallback)" header) instead of an error and a repair hint.
  Vector search resumes automatically once the chromadb segment becomes
  loadable again.
- `mempalace repair --mode legacy` → unchanged; works as before because
  it extracts via the fallback (which now works), then deletes and
  recreates the collection (which clears the divergence cache).
- Healthy collections (e.g., `mempalace_closets`, fresh palaces) skip the
  fallback entirely — no behavior change for non-divergent state.

### Tests run

```
python3 -m mempalace wake-up      # exit 0, was exit 139
python3 -m mempalace status       # exit 0, was exit 139
python3 -m mempalace search "..." # clean error with repair hint, was exit 139
python3 -m mempalace repair-status # already worked; still works
```

Layer2 with `where={"wing": "freighthero"}` and `$and` filters return
correct rows. Healthy closets collection still uses `ChromaCollection`
(verified via type assertion).

# mempalace fix: autofix surfaces stuck HNSW segment writer + opt-in rebuild

## Failure observed

```
$ mempalace diary write ...
InternalError: Failed to apply logs to the hnsw segment writer
$ mempalace add-drawer ...
InternalError: Failed to apply logs to the hnsw segment writer
```

The chromadb embeddings_queue keeps advancing (`seq_id` grows on every
new write) but every segment's `max_seq_id.seq_id` is frozen at its
pre-failure value, so the gap between the two grows monotonically.
`status` continued to look healthy because the existing autofix probes
(`orphaned_queue_rows`, `orphaned_max_seq_id_rows`, `poisoned_max_seq_id`,
`partial_hnsw_segments`) all check different shapes of corruption.

The recovery path already existed (`mempalace repair --mode from-sqlite`,
catalogued in `repair.py:1024`) but the operator had to know to run it.

## Fix

Add a "degraded HNSW writer" detector to autofix that compares
`embeddings_queue.MAX(seq_id)` against `max_seq_id.MAX(seq_id)` and
flags the gap when it crosses a 1000-row floor. Detection is read-only
and runs on every autofix call so `status` surfaces the degradation.
The applier is gated behind a new `allow_heavy=True` parameter because
the recovery action (`rebuild_from_sqlite`) re-embeds every drawer and
takes 30-60 minutes on a large palace — not safe to run silently from
`status`.

A new CLI mode `mempalace repair --mode hnsw-auto` opts into the
applier. It runs the probe in dry-run first, prints what would be
rebuilt, gates on `confirm_destructive_action` (same prompt
`from-sqlite` uses), then calls `auto_fix_palace(allow_heavy=True)`.

### Files

- **MODIFIED** `mempalace/autofix.py`
  - Module docstring updated: bounded contract now reads "*unless* the
    fixer is explicitly marked heavy and the caller passes
    `allow_heavy=True`."
  - `auto_fix_palace` gains an `allow_heavy: bool = False` parameter.
    Default preserves the existing `status`/`miner` invocation
    semantics (no behavior change for current callers).
  - `_probe_degraded_hnsw_writer` — new filesystem/SQLite probe. Reads
    `MAX(seq_id)` from `embeddings_queue` and `max_seq_id` via a
    read-only sqlite URI, coerces BLOB seq_ids (for palaces touched by
    the legacy 0.6.x shim before the poisoned-max-seq-id repair has
    run), bails on healthy gap / missing queue / missing
    `chroma.sqlite3`. Applier lazily imports
    `repair.rebuild_from_sqlite` so `status` calls don't pull in
    chromadb just to run the probe.
  - New probe block wired into `auto_fix_palace` after
    `partial_hnsw_segments`. Probe always appends a report; applier is
    only added to `pending` when `allow_heavy=True`. When detected but
    not allowed, the report's `detail` is suffixed with the recovery
    command hint so the operator sees it in `status` output.

- **MODIFIED** `mempalace/cli.py`
  - `--mode` choices for `mempalace repair` extended with `hnsw-auto`.
    Help text describes the trigger (stuck segment writer reported by
    `status`).
  - `cmd_repair` dispatch branch for `hnsw-auto`:
    1. Run `auto_fix_palace(palace_path, apply=False, allow_heavy=True)`
       to preview the probe result.
    2. If `degraded_hnsw_writer` not detected, print "Nothing to do" and
       return.
    3. Else gate on `confirm_destructive_action` (mirrors `from-sqlite`).
    4. Run `auto_fix_palace(palace_path, apply=True, allow_heavy=True)`
       and render the summary.
    5. Exit non-zero if the heavy applier captured an error so
       unattended callers (CI, cron) can detect partial recovery.

### Behavior after fix

- `mempalace status` → existing autofix output gains a
  `[detected] degraded_hnsw_writer: embeddings_queue is N entries
  ahead of applied max_seq_id (...) — run `mempalace repair --mode
  hnsw-auto` to rebuild` line when the gap exceeds 1000. No silent
  rebuild — `status` keeps its fast/bounded contract.
- `mempalace repair --mode hnsw-auto` → previews the gap, prompts for
  confirmation, then archives the existing palace dir to
  `<palace>.pre-rebuild-<timestamp>` and rebuilds from sqlite via the
  documented `rebuild_from_sqlite` recovery path.
- Healthy palaces (gap < 1000) → no detection, no behavior change.
- Existing autofix callers (`miner.status`, anything passing only
  `palace_path`) → unchanged because `allow_heavy` defaults to `False`.

### Tests run

In-tree smoke tests against a stub SQLite that mirrors the chromadb
schema autofix touches:

```
healthy gap=10                       → applier=None, detected=False
stuck gap=5010                       → detected=True, rows=5010, applier present
empty embeddings_queue               → applier=None, detected=False
allow_heavy=False (default)          → detected=True, applied=False
                                       detail suffixed with recovery command
allow_heavy=True + apply=False       → detected=True, applied=False (dry-run)
allow_heavy=True + apply=True        → applier runs; rebuild_from_sqlite errors
                                       on the stub schema are captured into
                                       report.error (auto_fix_palace contract
                                       "never raises" preserved)
```

# mempalace fix: autofix also detects silent HNSW index divergence

## Failure observed

```
$ mempalace repair-status
  [drawers]
    sqlite count:   274,199
    hnsw count:     27,121
    divergence:     247,078
    status:         DIVERGED
    note:           HNSW index holds 27,121 elements but sqlite has
                    274,199 embeddings — 247,078 drawers (90%) are
                    invisible to vector search. Run `mempalace repair`
                    to rebuild.
$ mempalace repair --mode hnsw-auto
  No degraded HNSW writer detected — nothing to do.
```

`mempalace_status` (MCP) returns `vector_disabled: true` with the same
divergence message, so callers correctly fall back to BM25 — but the
"auto" repair mode that exists for exactly this kind of recovery
refuses to act because the on-disk fingerprint is different from the
stuck-writer shape.

## Root cause

`_probe_degraded_hnsw_writer` only checks one shape: the
`embeddings_queue.MAX(seq_id) >> max_seq_id.MAX(seq_id)` gap that
appears when the segment writer panics mid-flush. The other recovery-
eligible shape — HNSW segment files containing fewer elements than
`chroma.sqlite3.embeddings` even with a fully-drained queue — has its
own detector (`backends/chroma.py::hnsw_capacity_status`, used by
`status` and `repair-status` to set `vector_disabled`) but the signal
never made it into the autofix queue. Operators had to know to switch
modes manually (`--mode from-sqlite --archive-existing`).

## Fix

Add `_probe_diverged_hnsw_index` to `mempalace/autofix.py` and wire it
into the same heavy-probe loop as the stuck-writer probe. The new probe
calls `hnsw_capacity_status` for both first-class collections (drawers
+ closets) and reports detection when either flips to `diverged`. The
applier is the same `rebuild_from_sqlite(archive_existing_dest=True)`
used for the stuck-writer fix — one rebuild covers every recoverable
collection in a single pass.

The `--mode hnsw-auto` CLI branch now treats both probes as recovery-
eligible: it gates on any detected heavy report, runs the same
`confirm_destructive_action` prompt, and exits non-zero if any heavy
applier captures an error.

### Files

- **MODIFIED** `mempalace/autofix.py`
  - The single `degraded_hnsw_writer` block is generalized into a
    `heavy_probes` tuple so adding a third heavy probe in the future is
    a one-line change. Per-probe `try`/`except` semantics preserved.
  - `_probe_diverged_hnsw_index` — new heavy probe. Lazy-imports
    `hnsw_capacity_status` and `CLOSETS_COLLECTION_NAME` to keep
    `autofix`'s import surface minimal on healthy-path calls. Reports
    `detected=True` when either collection's HNSW count trails its
    sqlite count past the threshold `hnsw_capacity_status` already
    enforces. Detail line lists each diverged collection so the
    operator sees which one (or both) is affected. Applier delegates to
    `repair.rebuild_from_sqlite` with `archive_existing_dest=True`.

- **MODIFIED** `mempalace/cli.py`
  - `--mode hnsw-auto` dispatch branch checks any report in
    `HEAVY_PROBE_NAMES = ("degraded_hnsw_writer", "diverged_hnsw_index")`
    instead of the single `degraded_hnsw_writer` name. "Nothing to do"
    message updated to mention both shapes. Non-zero exit on error
    extended the same way.
  - `--mode` argparse help text updated to mention the silent-
    divergence shape so `--help` reflects the broader trigger.

- **MODIFIED** `mempalace/miner.py`
  - `status` docstring updated to note that heavy operations are now
    detected for both fingerprints (stuck writer + index divergence)
    while preserving the no-silent-rebuild contract.

### Behavior after fix

- `mempalace status` / `mempalace_status` (MCP) → autofix output now
  also shows
  `[detected] diverged_hnsw_index: mempalace_drawers: HNSW N / sqlite M
  (K drawers, P% missing) — run \`mempalace repair --mode hnsw-auto\` to
  rebuild` whenever `hnsw_capacity_status` flips a collection to
  diverged. Vector search continues to fall back to BM25 (existing
  behavior) — the new line just makes the recovery command discoverable
  from the same place the divergence is reported.
- `mempalace repair --mode hnsw-auto` → fires on either stuck writer or
  silent divergence (or both). Same archive-then-rebuild flow. Operators
  no longer need to remember to switch to `--mode from-sqlite
  --archive-existing` when the writer is healthy but the index is
  silently undersized.
- Healthy palaces (both probes report no divergence) → message reads
  "No degraded HNSW writer or index divergence detected — nothing to
  do." No behavior change.
- Existing autofix callers (`miner.status`, anything passing only
  `palace_path`) → unchanged because `allow_heavy` still defaults to
  `False`. Detection is surfaced; rebuild is never auto-triggered.
