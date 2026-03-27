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

## Typical Validation Flow

1. Login with a valid user.
2. Select an authorized project.
3. Load queue items and apply filters when needed.
4. Listen to selected audio and submit validation status.
5. Resolve conflicts when concurrent updates occur.
6. Export or inspect project validation report.

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

## Troubleshooting

### The app does not start locally

1. Confirm Python 3.11+ is active.
2. Reinstall dependencies with `pip install -r requirements.txt`.
3. Run `python app.py` from the repository root.

### Port 7860 is already in use

Stop the existing process using that port, or launch after setting a different `GRADIO_SERVER_PORT` environment variable.

### Login works but no project appears

Your user likely has no project assignment. Add project access through the admin flow.

### Audio does not load for a detection

Check that:

1. `audio_id` exists and is valid for the selected project.
2. The dataset repository is reachable.
3. You have permission to read the project dataset.

### I get conflict messages while validating

This means another validator updated the same detection first. Refresh and reapply your decision on the newest version.
