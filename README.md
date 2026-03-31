---
title: BirdNET-Validator-App
emoji: "🐦"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.23.1"
python_version: "3.11"
app_file: app.py
pinned: false
---

# BirdNET-Validator-App

BirdNET-Validator-App is a Gradio application for human validation of BirdNET detections in multi-project workflows.

It provides a practical review interface with authentication, project-level authorization, paginated queues, on-demand audio, and append-only validation history.

## Objective

This project focuses on high-confidence validation operations by combining:

- controlled access by user and project
- rapid decision flow for detections
- auditability through append-only events
- conflict-aware updates for concurrent validators

## Key Usability Features

1. Multi-project login with role-based access
2. Project selection for authorized datasets
3. Detection queue with filters and pagination
4. Audio loading on demand with ephemeral cache
5. Fast validation actions (positive, negative, uncertain, skip)
6. Concurrency-safe writes with optimistic lock feedback
7. Conflict resolution support and validation reporting

## Quick Start (Local)

1. Create and activate a Python 3.11+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python app.py
```

Default port is 7860.

Optional runtime configuration:

- `BIRDNET_DETECTIONS_FILE`: path to JSON file with detections grouped by project slug.
- `BIRDNET_VALIDATIONS_DIR`: custom directory for append-only validation events and snapshot files.
- `BIRDNET_PAGE_SIZE`: queue page size (default: `25`).
- `BIRDNET_PROJECTS_FILE`: JSON file with project catalog used at app startup.
- `BIRDNET_USER_ACCESS_FILE`: JSON file mapping users to project roles (`admin`/`validator`).
- `BIRDNET_ENABLE_DEMO_BOOTSTRAP`: set to `true` only for local/demo mode to load built-in sample users/projects.

JSON seed format examples:

```json
{
	"kenya-2024": [
		{
			"detection_key": "0000000000001001",
			"audio_id": "audio_1001",
			"scientific_name": "Cyanocorax cyanopogon",
			"confidence": 0.91,
			"start_time": 0.0,
			"end_time": 1.0
		}
	]
}
```

Projects bootstrap file example (`BIRDNET_PROJECTS_FILE`):

```json
[
  {
    "project_slug": "kenya-2024",
    "name": "Kenya Survey 2024",
    "dataset_repo_id": "org/kenya-2024-dataset",
    "active": true
  }
]
```

User access bootstrap file example (`BIRDNET_USER_ACCESS_FILE`):

```json
{
  "admin_user": {
    "kenya-2024": "admin"
  },
  "validator_a": {
    "kenya-2024": "validator"
  }
}
```

## Typical Validation Flow

1. Login with a valid user.
2. Select an authorized project.
3. Load queue items and apply filters when needed.
4. Listen to selected audio and submit validation status.
5. Resolve conflicts when concurrent updates occur.
6. Export or inspect project validation report.

## Project CLI (Bootstrap Utilities)

The repository now includes a basic project bootstrap CLI:

```bash
python -m src.cli.project_cli --help
```

Available commands:

- `create-project`: adds a project entry to the projects bootstrap JSON.
- `init-dataset`: creates the initial local dataset folder structure.
- `build-index`: builds a confidence-sorted index from `detections.jsonl`.
- `verify-project`: verifies project config and required local scaffold files.
  - Use `--dry-run` to print findings without failing the command exit code.

Example flow:

```bash
python -m src.cli.project_cli create-project \
  --projects-file docs/spaces/projects.sample.json \
  --user-access-file docs/spaces/user_access.sample.json \
  --slug amazonia-2026 \
  --name "Amazonia 2026" \
  --dataset-repo-id birdnet/amazonia-2026 \
  --owner admin_user

python -m src.cli.project_cli init-dataset \
  --dataset-root data/projects \
  --slug amazonia-2026 \
  --dataset-repo-id birdnet/amazonia-2026 \
  --name "Amazonia 2026"

python -m src.cli.project_cli build-index \
  --dataset-root data/projects \
  --slug amazonia-2026

python -m src.cli.project_cli verify-project \
  --projects-file docs/spaces/projects.sample.json \
  --dataset-root data/projects \
  --slug amazonia-2026

python -m src.cli.project_cli verify-project \
  --projects-file docs/spaces/projects.sample.json \
  --dataset-root data/projects \
  --slug amazonia-2026 \
  --dry-run
```

## Repository Structure

- `app.py`: validator app entrypoint
- `src/auth`: authentication/session and ACL logic
- `src/domain`: domain models
- `src/repositories`: in-memory and append-only persistence
- `src/services`: queue, validation, and audio fetch services
- `src/ui`: Gradio interface composition and callbacks
- `tests`: unit and integration test suites

## Project Status

Validator workflow is active and under continuous development, with emphasis on reliability, auditability, and operator productivity.

## FAQ

### Is this app intended for production use?

Yes, for validator workflows. It is already structured for multi-project access control, audit-friendly writes, and collaborative validation.

### Does the app preload all audio files?

No. Audio is fetched on demand for the selected detection, helping keep memory and network usage under control.

### Can multiple validators work at the same time?

Yes. The validation flow uses optimistic concurrency checks to detect and handle conflicting updates.

### Can this run on Hugging Face Spaces?

Yes. The project is designed for Gradio deployment on Hugging Face Spaces.

## Deploy on Hugging Face Spaces

1. Create a new Space with:
- SDK: `Gradio`
- Python: `3.11`

2. Push this repository to the Space.

3. Configure Variables/Secrets in the Space settings.

Required for production bootstrap:
- `BIRDNET_PROJECTS_FILE` (example: `docs/spaces/projects.sample.json`)
- `BIRDNET_USER_ACCESS_FILE` (example: `docs/spaces/user_access.sample.json`)

Optional runtime settings:
- `BIRDNET_DETECTIONS_FILE` (seed detections JSON)
- `BIRDNET_VALIDATIONS_DIR` (default uses temp directory)
- `BIRDNET_PAGE_SIZE` (default `25`)
- `BIRDNET_ENABLE_DEMO_BOOTSTRAP` (`false` in production)

4. For first smoke test only, you may temporarily set:
- `BIRDNET_ENABLE_DEMO_BOOTSTRAP=true`

Then log in with one of the demo users:
- `admin_user`
- `demo_user`
- `validator_demo`

5. After validation, switch to production bootstrap:
- Set `BIRDNET_ENABLE_DEMO_BOOTSTRAP=false`
- Provide real `BIRDNET_PROJECTS_FILE` and `BIRDNET_USER_ACCESS_FILE`

Notes:
- The app entrypoint reads `PORT` automatically in Spaces.
- Keep user/project bootstrap files private if they contain sensitive assignments.

## Troubleshooting

### The app does not start locally

1. Confirm Python 3.11+ is active.
2. Reinstall dependencies with `pip install -r requirements.txt`.
3. Run `python app.py` from the repository root.

### Port 7860 is already in use

Stop the existing process using that port, or launch after setting a different `GRADIO_SERVER_PORT` environment variable.

### Login works but no project appears

Your user likely has no project assignment. Verify `BIRDNET_PROJECTS_FILE` and `BIRDNET_USER_ACCESS_FILE`, or add project access through the admin flow.

### Seed warning appears in Validation tab

If you see a seed warning banner:

1. Verify `BIRDNET_DETECTIONS_FILE` points to an existing file.
2. Confirm the file is valid UTF-8 JSON.
3. Confirm each project maps to a list of detections, or each list item includes `project_slug`.
4. If needed, unset `BIRDNET_DETECTIONS_FILE` to fall back to default demo detections.

### Audio does not load for a detection

Check that:

1. `audio_id` exists and is valid for the selected project.
2. The dataset repository is reachable.
3. You have permission to read the project dataset.

### I get conflict messages while validating

This means another validator updated the same detection first. Refresh and reapply your decision on the newest version.

## Recent Updates

- Integrated Validation tab flow in the multi-project app (no placeholder stage).
- Added project-scoped queue badge (`Queue: N`) and improved queue context feedback.
- Added seed-file warning banner with actionable remediation guidance in Validation.
- Standardized all user-facing UI text and feedback messages in English.
- Updated authentication/session timestamps to timezone-aware UTC handling.
