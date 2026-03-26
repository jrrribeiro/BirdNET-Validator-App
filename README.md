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
