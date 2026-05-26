# HF Trusted-Team Private Storage Delivery Plan

Status: immediate delivery architecture, implementation in progress on
`feature/project-state-security-privacy-review`
Created: 2026-05-26
Purpose: enable real validation by a small trusted research group while
protecting unpublished material from external access

## Decision Summary

For the immediate research workflow, all durable project data will live in
private Hugging Face resources under a dedicated project namespace:

```text
project organization namespace/
  audio-dataset                 private Dataset repository
  audio-dataset_state           private Dataset repository for manifest/ACL/invites
  audio-dataset_validation_state private Storage Bucket for validation events/snapshots
```

The working permission model is intentionally broad:

1. the project administrator owns or administers the namespace;
2. each trusted validator is granted Hugging Face `write` membership in the
   project organization;
3. each user signs in using their own Hugging Face authorization;
4. the app writes validation decisions to the private Bucket using the current
   validator's token, never a shared administrator token; and
5. users outside the organization cannot read the private dataset or state.

This is not the public distribution architecture. Trusted validators can
technically modify resources within the project organization outside the app.
That trade-off is explicitly accepted for a small trained research team.

## Why This Route Is Appropriate Now

The project needs to begin real validation before the BYO Supabase integration
can be researched and implemented. This route has already produced the key
permission proof:

| Actor | Organization role | Read private Bucket | Write private Bucket |
|---|---|---:|---:|
| Administrator | `admin` | passed | passed |
| Trusted validator | `write` | passed | passed |
| Contributor | `contributor` | passed | failed (`403`) |

Therefore, the free and currently demonstrated HF-only workflow requires
trusted validators to be `write` members of an isolated project organization.

## Current Branch Implementation Status

Implemented in `feature/project-state-security-privacy-review`:

1. explicit `BIRDNET_HF_TRUSTED_TEAM_MODE_ENABLED` runtime switch, disabled by
   default for existing deployments;
2. automatic creation of a private validation Bucket during new project setup
   in trusted-team mode;
3. automatic private `_state` repository creation with the Bucket reference in
   the project manifest;
4. rejection of public source datasets, personal namespaces, and app-private
   visibility in trusted-team mode;
5. no persisted shared project token for newly created Bucket-backed projects;
6. validation routing through the signed-in actor's HF token with no
   administrator-token fallback;
7. HF-mode health status and an Admin warning explaining organization `write`
   membership; and
8. a private Bucket access probe that reads, writes, verifies, and removes a
   diagnostic marker using the signed-in collaborator's own token; and
9. unit tests for configuration, private-dataset enforcement, organization
   enforcement, token removal, Bucket access/routing, conflict/reconciliation,
   state recovery, and UI creation.

Recovery through **Connect Existing State** also refuses manifests that do not
declare Bucket storage or no longer identify a compatible private
organization-owned collaborative project.

Remaining before merge with `main`:

1. push/deploy this branch to the Space;
2. configure the runtime variables/secrets;
3. run the two-account private organization test described in the merge gate;
4. confirm rebuild recovery through `BIRDNET_HF_PROJECT_STATE_REPOS`; and
5. confirm exports after real Bucket-backed validations.

Automated verification completed on 2026-05-26:

1. `python -m pytest -q`: `246 passed`;
2. trusted-team UI construction with
   `BIRDNET_HF_TRUSTED_TEAM_MODE_ENABLED=true`: passed; and
3. default UI construction without the new mode: passed; and
4. `git diff --check`: no patch whitespace errors.

## Security Boundary

### Protected Against

1. external users not granted organization membership;
2. accidental loss caused by Space rebuilds or lack of persistent disk;
3. disclosure of a shared administrator dataset token through the app;
4. high-frequency Git commit limits for individual validation actions, because
   validation events are written to a Bucket rather than a dataset repository.

### Not Protected Against

1. intentional or accidental modification by a trusted organization member
   with `write` permission;
2. a compromised member Hugging Face token;
3. project-to-project privilege spillover if multiple confidential projects
   are placed in the same organization.

Mitigation: create a separate HF organization for each confidential project or
for each group of projects whose validators may safely share broad access.

## Resource Responsibilities

| Resource | Purpose | Write pattern | Authority |
|---|---|---|---|
| Private audio dataset | Audio and original detection index | Uploader only before/during ingest | Project admin/uploader |
| Private `_state` dataset | `project.json`, `acl.json`, `invites.json` | Infrequent admin operations | App admin flow |
| Private validation Bucket | events, snapshots, archives | Frequent validator writes | Current signed-in HF user |
| Space runtime/local files | Cache only | Ephemeral | Never authoritative |

The audio dataset must be private before a real project is created in this
mode. The source dataset should be treated as immutable once active validation
begins, other than controlled uploader completion.

## State Layout

### Private `_state` Repository

```text
project.json
acl.json
invites.json
README.md
```

The repository contains project configuration and app-level access metadata.
It must not contain HF tokens.

### Private Validation Bucket

```text
metadata/project.json
snapshots/current.json
events/YYYYMMDD/<event-id>.json
archives/events/<archive-id>.jsonl
```

Rules:

1. individual decisions are append-oriented events;
2. snapshots accelerate queues and reports but are reconstructable;
3. compaction archives active events instead of discarding audit history;
4. parallel same-version decisions are surfaced as conflicts; and
5. exports join validation state with the source detection metadata.

## Application Configuration

The immediate mode is activated explicitly:

```text
BIRDNET_HF_TRUSTED_TEAM_MODE_ENABLED=true
BIRDNET_AUTH_MODE=hf_token
BIRDNET_HF_PROJECT_STATE_REPOS=project-org/audio-dataset_state
HF_TOKEN=<admin read/write token for startup recovery>
```

Notes:

1. `BIRDNET_HF_TRUSTED_TEAM_MODE_ENABLED=true` implies HF `_state` syncing and
   HF Bucket validation routing for projects created in the mode.
2. `BIRDNET_AUTH_MODE=hf_token` permits each team member to use their own
   verified token. OAuth-only Bucket capability has not been proven.
3. `HF_TOKEN` allows the Space to restore project configuration after a
   restart before any user has logged in. It belongs to the administrator of
   this specific deployment and must be stored only as a Space secret.
4. `BIRDNET_HF_PROJECT_STATE_REPOS` is updated after each project is created,
   so the Space can reload it after rebuilds.
5. Project tokens stored in application state are disabled for new
   trusted-team Bucket projects. Users supply their own authorization.
6. The hosted OAuth button requests read access only in this interim mode;
   it does not replace the personal token required for the proven Bucket
   write workflow.

## Administrator Workflow

### Prerequisites

1. Create a dedicated Hugging Face organization for the confidential project.
2. Ensure the audio dataset exists in that namespace and is private.
3. Ensure the administrator token can read/write resources in the
   organization.
4. Configure the Space secrets/variables above.

### Create Project In The App

1. Log into the app with the administrator's own HF token.
2. Open **Admin > Project management**.
3. Enter:
   - project slug;
   - display name;
   - private dataset repo id, for example `project-org/audio-dataset`;
   - app visibility `collaborative` for a team project.
4. Select **Create Project**.
5. The app verifies that the source dataset is private, organization-owned,
   and uses app visibility `collaborative`.
6. The app creates:
   - `project-org/audio-dataset_validation_state` as a private Bucket;
   - `project-org/audio-dataset_state` as a private state repository.
7. Copy the new `_state` repo id into the Space variable
   `BIRDNET_HF_PROJECT_STATE_REPOS`, comma-separated with existing projects.
8. Restart/rebuild the Space and confirm the project is restored.

### Add Trusted Validators

1. Add the validator to the dedicated project organization in Hugging Face
   with `write` role.
2. Assign or invite the same HF username in the app to retain workflow roles
   and UI filtering.
3. Validator logs in using their own HF token.
4. Validator opens **Projects > Private project storage access** and runs the
   check, which writes and removes a small diagnostic marker without changing
   validation records.
5. Validator verifies that the private audio opens and submits a test
   validation.

Both membership layers are intentionally required in this interim model:
Hugging Face controls actual private resource access, while app roles control
the validation interface.

## Implementation Work Packages

### Package 1 - Explicit Runtime Mode

Implementation:

1. Add `BIRDNET_HF_TRUSTED_TEAM_MODE_ENABLED`.
2. Make the mode enable project-state synchronization and Bucket validation
   routing without changing existing deployments by default.
3. Display active private HF storage status in Settings and app header.

Tests:

1. environment flag parsing;
2. default disabled behavior;
3. UI construction smoke test.

### Package 2 - Secure Project Provisioning

Implementation:

1. During project creation, require an organization-owned private source
   dataset and app visibility `collaborative`.
2. Initialize the private Bucket in the dataset namespace.
3. Store Bucket reference in the `_state` manifest.
4. Initialize the private `_state` repository.
5. Do not persist setup/project tokens for Bucket projects.
6. Refuse automatic overwrite if either store contains unrelated content.

Tests:

1. private dataset accepted and public dataset rejected;
2. Bucket initialization and manifest linkage;
3. existing safe state reuse and unsafe content refusal;
4. setup token does not appear in serialized project state.

### Package 3 - Trusted Validator Writes

Implementation:

1. Route validation reads/writes to the Bucket for projects declaring it.
2. Use the current signed-in user's HF token for writes.
3. Never fall back to the administrator token for a validator write.
4. Give clear access errors when organization permission is missing.

Tests:

1. actor token routing;
2. no fallback to admin token;
3. disabled-backend hard failure;
4. two-user Space proof with private audio and Bucket.

### Package 4 - Recovery And Audit

Implementation:

1. Reload project metadata from configured `_state` repositories on startup.
2. Keep events append-oriented and snapshots reconstructable.
3. Preserve conflict reporting and archive compaction.
4. Expose CSV/XLSX exports from reconciled current state.

Tests:

1. rebuild Space and restore project/ACL;
2. restore snapshot from events;
3. simultaneous different-audio validation;
4. same-audio conflict;
5. export before and after restart.

### Package 5 - Operational Guardrails

Implementation:

1. Add visible wording that trusted-team mode requires HF organization
   membership separately from app invitations.
2. Provide a collaborator-run Bucket read/write diagnostic that removes its
   own marker and does not create a validation decision.
3. Warn when no `_state` repository is configured for restart recovery.
4. Document backup/export cadence.
5. Document token revocation after team changes.
6. Record that organization isolation is the privacy boundary.

## Merge Gate For `main`

The branch may be merged only after:

1. the existing automated test suite passes;
2. a private test dataset in a dedicated organization is loaded in the Space;
3. the administrator creates a project and sees both private state resources;
4. the Space is rebuilt and restores the project from `_state`;
5. administrator and second trusted account pass **Private project storage
   access** using their own tokens;
6. a second trusted account with organization `write` role loads private
   audio and writes a validation using its own token;
7. an account outside the organization cannot load audio or validation state;
8. twenty rapid validations persist after refresh/rebuild;
9. a CSV and XLSX export contain the expected validation decisions; and
10. the administrator keeps a source backup of audio/index data.

## Known Limitations

1. Creating organizations and managing membership remain Hugging Face web UI
   steps in the free supported workflow investigated to date.
2. A trusted validator receives broad write permission in that project's
   organization.
3. The startup recovery secret is acceptable for the administrator's immediate
   private deployment, but is not the distributed public-product model.
4. OAuth-only access to private Buckets remains unproven; routine immediate
   testing uses each member's own verified HF token.

## Transition To Distribution Architecture

When BYO Supabase with Edge Function support is proven:

1. retain the private HF audio dataset as source storage;
2. export/replay Bucket validation events into the administrator-owned
   Supabase database;
3. replace organization-wide write membership with app-issued validation
   authorization; and
4. archive the HF Bucket as a historical backup or retain it read-only.

## Official References

1. Hugging Face Storage Buckets:
   <https://huggingface.co/docs/hub/storage-buckets>
2. Hugging Face dataset repositories:
   <https://huggingface.co/docs/hub/datasets-overview>
3. Hugging Face organizations:
   <https://huggingface.co/docs/hub/organizations>
4. Hugging Face OAuth:
   <https://huggingface.co/docs/hub/oauth>
