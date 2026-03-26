# HF Segments Ingestion Plan

## Objective
Build a dedicated ingestion tool (separate from validation UI) that accepts:
1. A BirdNET detections CSV
2. A root segments folder (including subfolders)

Then uploads both audio segments and a normalized metadata index to a Hugging Face dataset repo.

## Why This Comes Next
- Validation UI is already stable for auth/admin/ACL.
- Real operations depend on real data availability.
- Ingestion first avoids UI rework over mock assumptions.

## Real Dataset Observations (from local reference)
- Segments root folders (species folders): 293
- Segment files: 146562
- Extension profile: `.wav` only (plus one `_progress_log.txt`)
- Filename pattern sample:
  - `Accipiter striatus/Catim_LS-1500_U10_20250221_060600_0.0-3.0s_85%.wav`
  - `Cyanocorax cyanopogon/Aiuab_LN-0500_U16_20260121_153400_42.0-45.0s_98%.wav`

## Input CSV (reference provided)
Observed columns include:
- `locality`
- `point`
- `date_folder`
- `source_file`
- `scientific_name`
- `common_name`
- `confidence`
- `start_time`
- `end_time`
- `exact_start`
- `exact_end`
- `min_freq`
- `max_freq`
- `box_source`
- `label`

## Proposed HF Dataset Layout
- `audio/segments/<scientific_name>/<segment_filename>.wav`
- `index/shards/*.parquet` (normalized row index for fast reads)
- `manifest.json` (schema/version/counters)
- `audit/ingestion-runs/<run_id>.json` (run report)
- `audit/unmatched/<run_id>.parquet` (rows not linked to files)

## Canonical Metadata Row (Normalized)
Required fields for each detection segment:
- `project_slug`
- `detection_key` (stable deterministic key)
- `audio_id` (source audio id, from `source_file` stem)
- `segment_filename`
- `segment_relpath`
- `scientific_name`
- `common_name`
- `confidence`
- `start_time`
- `end_time`
- `exact_start`
- `exact_end`
- `locality`
- `point`
- `date_folder`
- `min_freq`
- `max_freq`
- `box_source`
- `label`
- `source_file`
- `ingested_at`
- `schema_version`

Optional convenience fields:
- `segment_duration`
- `confidence_pct_rounded`
- `match_strategy`
- `match_score`

## Matching Strategy (CSV <-> Segment Files)
Primary join (deterministic):
1. species folder = `scientific_name`
2. segment filename starts with `source_file` stem
3. filename contains `<start_time>-<end_time>s`

Tolerance rules:
- Use decimal normalization (`3` vs `3.0`) for time token search.
- Confidence in filename (`85%`) is optional for join, used as tie-breaker only.

Tie-break policy:
- If more than one segment matches, pick exact time token first.
- If still ambiguous, pick closest confidence percentage.
- Persist ambiguity in audit file.

## Deterministic detection_key
Generate with stable hash to satisfy model constraints and dedupe safely:
- input: `project_slug|source_file|scientific_name|start_time|end_time`
- `detection_key = sha256(input).hexdigest()[:24]`

Benefits:
- deterministic across reruns
- longer than minimum length constraints
- independent from local filesystem ordering

## CLI Proposal (new command)
Add command to `cli/hf_dataset_cli.py`:
- `ingest-segments`

Arguments:
- `--project-slug`
- `--dataset-repo`
- `--detections-csv`
- `--segments-root`
- `--batch-size` (default 200)
- `--shard-size` (default 10000)
- `--dry-run`
- `--resume-state-file` (default `.ingest-segments-state.json`)
- `--max-retries` (default 3)
- `--retry-backoff-seconds` (default 1.0)

## Command Behavior
Dry-run mode:
- validates input schema
- computes matching stats
- reports would-upload counts
- writes no remote files

Execute mode:
1. ensure dataset scaffold
2. parse CSV and normalize rows
3. discover segments recursively
4. match rows to segment files
5. upload segment audio in batches with resume/retry
6. build/upload index shards from matched rows
7. update `manifest.json`
8. write audit artifacts

## Operational Report (stdout JSON)
Return key counters:
- `csv_rows_total`
- `segments_found_total`
- `matched_rows`
- `unmatched_rows`
- `ambiguous_rows`
- `uploaded_audio_now`
- `uploaded_audio_skipped_existing`
- `index_rows_written`
- `shards_written`
- `failed_uploads`
- `duration_seconds`

## Failure Policy
Hard-fail:
- missing required CSV columns
- unreadable segments root
- HF auth/repo unreachable

Soft-fail with audit:
- unmatched rows
- ambiguous matches
- partial audio upload failures

Exit code:
- `0` when no upload failures (unmatched allowed with warning)
- `1` when upload/index write failures occur

## Performance Notes
Given ~146k WAV files:
- keep file discovery cached in-memory map by species + source_file stem
- avoid O(N*M) matching scans
- process CSV in chunks where possible
- upload in batches and persist resume state after each batch

## Security and Separation
- Ingestion tool remains separate from validation UI.
- Validation app reads index/audio already published to HF.
- ACL remains project-scoped at validation stage.

## Implementation Phases
Phase 1 (MVP ingestion):
- `ingest-segments --dry-run`
- deterministic key generation
- CSV normalization + matching report

Phase 2 (publish):
- batched audio upload + resume/retry
- index shard generation and manifest update

Phase 3 (hardening):
- audit files for unmatched/ambiguous
- duplicate handling policy and idempotency checks
- integration tests with fixture CSV + synthetic segment tree

## Suggested Next Action
Implement Phase 1 first (`dry-run`) so we can validate matching quality on your real CSV and folder before any remote upload.
