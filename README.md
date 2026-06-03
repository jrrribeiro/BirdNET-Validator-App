---
title: BirdNET-Validator-App
emoji: "🐦"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.23.1"
python_version: "3.11"
app_file: app.py
hf_oauth: true
hf_oauth_scopes:
  - read-repos
pinned: false
---

# BirdNET Validator

BirdNET Validator is a Gradio app for collaborative review of BirdNET audio
segments stored in Hugging Face datasets. It is designed for research teams that
need authenticated validators, private project state, validation progress, and
downloadable scientific tables.

The uploader is maintained separately in `HF_Dataset_Uploader`. This repository
contains only the validator app.

## Current Architecture

- Hugging Face OAuth identifies the signed-in validator.
- Each project points to one Hugging Face dataset containing audio segments and
  metadata/index files.
- Project metadata and access rules live in a private companion dataset repo
  named with the `_state` suffix.
- Validation events and current validation snapshots are stored in private
  Hugging Face storage controlled by the app backend.
- Project access, invitations, validation status, progress, and exports are
  managed from the app UI.

## Main Features

- Secure Hugging Face sign-in with app-managed project access.
- Admin project creation, state connection, token management, invitations, and
  safe deletion.
- Validation queue organized by species, confidence, and pending status.
- Audio playback, spectrogram preview, keyboard shortcuts, and correction rules.
- Required corrected species when rejecting a segment.
- Progress dashboards by species, validator, and recent activity.
- CSV/XLSX export combining the original detection table with validation fields.

## Required Space Secret

For the current private HF storage mode, configure this secret in the Hugging
Face Space:

```text
BIRDNET_HF_STORAGE_TOKEN
```

Use a Hugging Face token from the Space/project owner account with enough
permission to create, read, write, and delete the private project state repos and
validation storage used by the app.

## Useful Optional Settings

```text
BIRDNET_PAGE_SIZE=10
BIRDNET_VALIDATOR_HOST=0.0.0.0
BIRDNET_VALIDATOR_PORT=7860
BIRDNET_INVITE_EMAIL_MODE=off
BIRDNET_INVITE_FROM_EMAIL=
BIRDNET_INVITE_REPLY_TO=
```

EmailJS settings are optional and only needed when email invitations are enabled:

```text
BIRDNET_EMAILJS_SERVICE_ID=
BIRDNET_EMAILJS_TEMPLATE_ID=
BIRDNET_EMAILJS_PUBLIC_KEY=
BIRDNET_EMAILJS_PRIVATE_KEY=
```

## Local Development

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Verification

Run the automated tests:

```bash
pytest -q
```

Run a lightweight deployment check:

```bash
python scripts/check_deployment.py
```

## Repository Layout

```text
app.py                         Gradio app entry point
src/auth/                      Hugging Face identity and access helpers
src/cache/                     Ephemeral audio/cache utilities
src/config/                    Runtime configuration
src/domain/                    Shared data models
src/repositories/              Project state and validation persistence
src/services/                  Queue, audio, validation, deletion, email services
src/ui/                        Gradio screens, components, and theme
scripts/check_deployment.py    Pre-deployment smoke check
tests/                         Unit and integration tests
```

## Deployment

The GitHub workflow `sync-hf-space.yml` uploads `main` to the Hugging Face Space.
Set the GitHub Actions secret `HF_TOKEN` with write access to the target Space.

The CI workflow runs tests on every push and pull request to `main`.
