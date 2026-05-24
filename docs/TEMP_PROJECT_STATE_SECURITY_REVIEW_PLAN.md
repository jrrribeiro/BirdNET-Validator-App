# Temporary Plan: Project State, Security, and Privacy Review

Status: planning document  
Created: 2026-05-21  
Scope: BirdNET Validator App project persistence, authentication, authorization, secret handling, and validation-state ownership

## Purpose

This document records the next structural review for the BirdNET Validator App after the Gradio UI and current Supabase-backed workflows reached a usable test state.

The next target is not a visual redesign. It is a security and persistence redesign:

1. Keep the shared Space lightweight and as close to stateless as practical.
2. Avoid making the maintainer's Supabase account the permanent storage and cost center for third-party projects.
3. Store each project's durable state under the project admin's Hugging Face account or organization when feasible.
4. Preserve privacy, auditability, collaboration, and recovery across Space redeploys and app updates.

This is a temporary engineering plan. It should be revisited before implementation and converted into issue-sized tasks once the target state backend is validated.

## Executive Decision

The recommended product direction is:

> Project audio data and validator state should be owned by the project admin or the admin's organization. The BirdNET Validator Space should authenticate users, load project state, and operate on state stores it does not centrally own.

The current Supabase backend should remain available during development and can remain an advanced backend option, but it should not be the only production architecture for a public multi-project app.

## Current Architecture Summary

### Runtime modes

The app currently supports:

1. Filesystem state
   - Project bootstrap files.
   - User access files.
   - Invite files.
   - Append-only validation files.
   - Suitable for local tests or Spaces with persistent storage.

2. Supabase state
   - Project records.
   - Project ACL assignments.
   - Pending invites.
   - Validation event log.
   - Current validation snapshot.
   - Suitable for current Spaces testing without persistent local storage.

### Current state surfaces

The following durable state exists or is planned to exist per project:

| State | Current storage | Sensitivity | Expected growth |
|---|---|---|---|
| Project metadata | filesystem or Supabase | low to medium | tiny |
| Dataset reference | filesystem or Supabase | low unless private metadata is sensitive | tiny |
| Project ACL | filesystem or Supabase | medium | small |
| Pending invites | filesystem or Supabase | medium, may include email | small |
| Project dataset token | filesystem or Supabase | high | tiny but risky |
| Validation events | filesystem or Supabase | medium | high |
| Current validation snapshot | filesystem or Supabase | medium | high |
| Derived progress reports | computed | low to medium | cacheable |

### Current useful properties

The app already has several good structural properties:

1. Roles are scoped per project.
2. Admin operations are checked in backend callbacks.
3. Private projects block collaborator invites.
4. Validation writes use append-only history plus current snapshots.
5. Optimistic conflict handling exists for concurrent validation writes.
6. Dataset index loading now prefers the known `HF_Dataset_Uploader` index path before expensive repository tree discovery.

## Risk Review

## 1. Authentication risk

### Current concern

Username-only login is convenient for demo and local testing, but it does not prove identity in production.

### Required direction

Production login must require a verified Hugging Face identity:

1. Hugging Face OAuth for Spaces as the preferred flow.
2. Token-backed `whoami` validation only as an explicit fallback or developer mode.
3. Username-only login restricted to demo/test mode.

### Acceptance criteria

1. A production user cannot claim another Hugging Face username by typing it.
2. Session identity is derived from an authenticated Hugging Face identity.
3. The app does not persist personal HF tokens in project state.

## 2. Secret storage risk

### Current concern

Projects can persist `dataset_token`. This can centralize admin tokens in the app's persistence layer.

### Required direction

Prefer tokenless project records:

1. Users access datasets and project state with their own authenticated HF permissions.
2. Project tokens are not stored by default.
3. Any unavoidable secret support is isolated behind explicit opt-in backend configuration, encrypted storage, least privilege, audit logs, and clear warnings.

### Acceptance criteria

1. The standard collaborative project flow does not require a stored project token.
2. Validators cannot read token values through the UI, logs, reports, state exports, or error messages.
3. Project-state backends have an explicit policy for whether they may persist secrets.

## 3. Central storage and quota risk

### Current concern

One maintainer-owned Supabase project currently stores all project metadata and validation state when Supabase mode is enabled.

### Consequences

1. Quota and possible future cost accumulate on the maintainer.
2. A single backend outage can affect all projects.
3. The maintainer becomes custodian of third-party ACL and validation data.
4. Growth is dominated by validation events, not by the project catalog.

### Required direction

Use admin-owned project stores for the community/default architecture.

## 4. Privacy and ownership risk

### Current concern

ACL, invite emails, project metadata, and validation decisions can reveal project membership and work activity.

### Required direction

1. Project admins own their durable project state.
2. State visibility follows project visibility and admin decisions.
3. Email invites are optional and must avoid overexposing private project details.
4. Logs should avoid tokens, email contents, and raw secrets.

## Target Architecture

## Core principle

The Space should not be the long-term owner of user projects.

Each project should reference:

1. An audio dataset.
2. A validator state store.
3. A schema version.
4. A security policy.

## Recommended project layout

### Audio dataset

Existing source of segments and uploader index:

```text
owner-or-org/audio-dataset
```

### Validator state store

New durable state owned by the admin or organization:

```text
owner-or-org/audio-dataset-validator-state
```

The first implementation spike should compare:

1. Hugging Face Storage Bucket state.
2. Companion Hugging Face dataset repository state.

The preferred option after the spike should become the default `HfProjectStateBackend`.

## Proposed project manifest

The project record should contain a non-secret manifest-like reference:

```json
{
  "schema_version": 1,
  "project_slug": "amazonia-2026",
  "project_name": "Amazonia 2026",
  "dataset_repo_id": "owner-or-org/amazonia-audio",
  "state_backend": "hf_project_store",
  "state_ref": "owner-or-org/amazonia-audio-validator-state",
  "owner_username": "owner-or-org-admin",
  "visibility": "collaborative"
}
```

This manifest must not contain a private token.

## Proposed durable state objects

The exact storage format may change after the backend spike, but state must be separable into:

```text
project.json
acl.json or acl table
invites.json or invites table
events/
snapshots/
reports/ optional derived cache
locks/ or reservations/ optional workflow coordination
```

### Minimum event record

Validation history must preserve:

1. Project identifier.
2. Detection key.
3. Status.
4. Validator identity.
5. Timestamp.
6. Corrected species if any.
7. Notes if any.
8. Expected and resulting version or equivalent conflict metadata.

## Backend Abstraction Plan

Create an explicit project-state interface before adding more persistence logic to `app_factory.py`.

## Required interfaces

At minimum separate:

1. `ProjectCatalogRepository`
2. `ProjectAccessRepository`
3. `ProjectInviteRepository`
4. `ValidationEventRepository`
5. `CurrentValidationRepository`

If a smaller initial abstraction is needed, begin with:

```python
class ProjectStateBackend:
    def load_project(...)
    def create_project(...)
    def load_acl(...)
    def save_acl(...)
    def load_invites(...)
    def save_invites(...)
    def append_validation_event(...)
    def load_current_snapshot(...)
    def save_current_snapshot(...)
```

## Planned implementations

| Backend | Role |
|---|---|
| Filesystem | local/demo/dev and offline tests |
| Supabase | current backend, advanced hosted DB mode |
| Hugging Face companion `_state` repository | recommended control-plane store for project manifest, ACL, invites and checkpoints |
| Hugging Face Storage Bucket | target high-frequency store for validation events and mutable snapshots |

## Authentication and Authorization Plan

## Production login policy

1. Enable Hugging Face OAuth flow in the Space.
2. Restrict requested scopes to the smallest set proven necessary.
3. Keep explicit token login only when OAuth does not cover a required flow.
4. Keep username-only login behind a demo/development flag.

## Project authorization policy

Authorization needs two layers:

1. App/project ACL
   - Admin.
   - Validator.
   - Optional future reviewer/auditor roles.

2. Hugging Face resource permissions
   - Dataset read permissions.
   - State store read/write permissions.

Both must pass for a protected operation.

## Role expectations

| Action | Admin | Validator |
|---|---|---|
| Read authorized project metadata | yes | yes |
| Read audio dataset if HF permissions allow | yes | yes |
| Validate assigned project | yes | yes |
| Read project progress | yes | yes or policy-controlled |
| Invite/remove users | yes | no |
| Change state store reference | yes | no |
| Delete/archive project state | yes with safeguards | no |

## Project Creation Flow

## Target flow

1. User authenticates with verified HF identity.
2. User selects an existing dataset or provides a dataset repo id.
3. App verifies dataset access.
4. User chooses the state backend.
5. Default flow creates or connects a state store owned by that user or organization.
6. App writes project manifest and initial ACL.
7. Creator becomes project admin.
8. Project becomes visible to the creator after successful state-store initialization.

## Failure policy

Project creation must be transactional enough to avoid half-created UI records:

1. If state store creation fails, project is not shown as ready.
2. If manifest write fails, state store is marked incomplete or cleaned up if safe.
3. The UI must show a recoverable status instead of silently creating broken projects.

## Validation Write Strategy Review

## Problem to solve

HF-backed object or repo storage is not automatically equivalent to a relational transactional database.

The spike must decide how to preserve:

1. Append-only audit history.
2. Current snapshot lookup speed.
3. Concurrency safety.
4. Progress dashboard speed.
5. Recovery after app restart.

## Candidate strategies

### Strategy A: event batches plus periodic snapshots

1. Append validation events in small batch objects.
2. Maintain compact snapshots by project.
3. Rebuild snapshot if needed from events.

Pros:
- Audit-friendly.
- Portable.
- Works with storage systems that like batch writes.

Risks:
- Needs conflict strategy.
- Snapshot updates need merge discipline.

### Strategy B: dataset commits with batched JSONL/Parquet files

Pros:
- Native HF repository history.
- Easy download/export.

Risks:
- Commit pressure.
- More latency.
- Git/LFS history is not ideal for frequent small writes.

### Strategy C: admin-owned database

Pros:
- Strongest concurrency and query ergonomics.

Risks:
- Setup burden.
- Not the default no-cost/no-maintainer-cost flow.

## Initial recommendation

Prototype Strategy A for the HF project store. Keep Supabase Strategy C as an advanced backend.

## Storage Growth and Cost Model

## Important observation

Project catalog size is negligible compared with validation history.

### Estimate categories

| Category | Expected size |
|---|---|
| Project manifest | KB |
| ACL and pending invites | KB to small MB |
| Current snapshot | grows with validated detection count |
| Validation events | grows with validation actions and revisions |
| Cached reports | optional, bounded |

## Required product behavior

1. State store belongs to admin or organization.
2. Project exports are supported.
3. Archiving is supported.
4. Retention policy is explicit for any derived cache.
5. Event history is not silently destroyed.

## Security Review Checklist

Implementation must address:

1. Verified identity only in production.
2. Least-privilege HF OAuth scopes.
3. No token values in logs.
4. No token values in Gradio component values after submit.
5. No token values in state exports by default.
6. Admin-only ACL mutation.
7. Admin-only state backend mutation.
8. Validator cannot inspect unauthorized project metadata.
9. State store reference cannot be used to escape project authorization.
10. Invite links and invite messages do not become bearer access tokens unless deliberately designed and protected.
11. Invite email addresses are minimized and covered by privacy documentation.
12. Project deletion/archive flow has confirmation and recovery policy.
13. Public datasets and private project state are handled as separate visibility decisions.

## Migration Plan

## Phase 0: freeze assumptions

Deliverables:

1. Confirm current production deployment mode.
2. Confirm whether the desired default is HF Bucket or companion dataset after a technical spike.
3. Confirm whether validation state may be public when audio datasets are public.

Exit criteria:

1. Written architecture decision.
2. One real project selected for migration testing.

## Phase 1: production auth hardening

Deliverables:

1. OAuth-first login path.
2. Development-only flag for username login.
3. Session identity tests.
4. Updated settings/status panel showing active auth mode without exposing secrets.

Tests:

1. User A cannot impersonate User B.
2. Invalid token cannot create a session.
3. OAuth identity maps to ACL identity.

## Phase 2: state abstraction refactor

Deliverables:

1. Move state persistence contracts out of UI callbacks.
2. Keep Supabase implementation passing existing tests.
3. Keep filesystem implementation for local/demo mode.
4. Add contract tests shared by each backend.

Tests:

1. Project create/load/archive.
2. ACL grant/revoke/load.
3. Invite create/accept/revoke/expire.
4. Validation append/snapshot/conflict.

## Phase 3: HF project-state backend spike

Deliverables:

1. Minimal backend for one project store.
2. State-store initialization flow.
3. Read/write validation proof.
4. Multi-user project permission proof.
5. Restart/redeploy persistence proof.

Questions to answer:

1. Bucket or companion dataset?
2. How are write permissions granted to validators?
3. Does the app write on behalf of each validator or via an admin-owned delegated path?
4. How are atomic version updates represented?
5. What write batching is necessary for good latency?

Exit criteria:

1. One real project can validate, reload, report progress, and survive app redeploy without Supabase.

### 2026-05-24 Architecture Decision Update: high-frequency validation writes

The companion private dataset repository remains appropriate for project manifest,
ACL, invite metadata, durable exports, and recovery anchors. It must not remain
the primary write target for every validation click at production scale.

Reason:

1. An expert validator may submit one validation approximately every three seconds.
2. Several validators can work simultaneously on the same project.
3. A Git-backed dataset commit for every action produces avoidable commit pressure,
   history growth, and latency.
4. Hugging Face documentation now recommends Storage Buckets for mutable data and
   high-frequency small writes, while repositories remain appropriate for versioned
   artifacts and documentation.

Target storage split:

| State | Target store | Write profile |
|---|---|---|
| Project manifest, ACL and invite policy | private companion `_state` repository | infrequent administrative commits |
| Validation events and current snapshot | private admin-owned Storage Bucket | frequent mutable writes |
| Periodic validation export/checkpoint | private companion `_state` repository or downloadable export | infrequent audited checkpoint |

Token and permission policy:

1. Users authenticate with their own Hugging Face OAuth identity.
2. Tokens remain session-scoped and are never serialized into project state.
3. The project admin creates/owns the private state resources through the app.
4. A multi-user permission proof is required before enabling bucket-backed
   validation by default; validators must never depend on a shared admin token.
5. If Hugging Face resource permissions cannot grant validator writes safely, an
   alternative delegated write path must be designed before production rollout.

## Phase 4: project onboarding flow

Deliverables:

1. Create project wizard.
2. Connect existing project store wizard.
3. Ownership and permission validation.
4. Recovery UI for incomplete state stores.
5. Optional export manifest.

## Phase 5: migration tools

Deliverables:

1. Export Supabase project state.
2. Import into HF project store.
3. Verification report comparing:
   - project metadata;
   - ACL;
   - invites if retained;
   - current snapshot;
   - event count.
4. Rollback plan.

## Phase 6: documentation and production defaults

Deliverables:

1. Privacy and storage ownership explanation.
2. Admin onboarding guide.
3. Validator access guide.
4. Backend selection guide.
5. Security caveats for public/private datasets and state stores.
6. Updated deployment docs that do not present maintainer-owned Supabase as the only durable option.

## Testing Strategy

## Unit tests

1. Auth mode gates.
2. ACL checks.
3. Invite lifecycle.
4. Backend contracts.
5. Secret redaction.
6. Manifest validation.
7. Migration transforms.

## Integration tests

1. Filesystem backend.
2. Supabase backend with mocked REST boundary.
3. HF project-state backend with mocked Hub/storage boundary.
4. Dataset index load versus state load.

## Manual end-to-end tests

1. Admin creates project from HF dataset.
2. App initializes project-owned state.
3. Admin invites validator.
4. Validator accepts access.
5. Validator validates audio.
6. Admin sees progress.
7. Project survives app restart and deploy.
8. Unauthorized user cannot see or mutate project.
9. Private project cannot be joined without policy approval.

## Load and scale tests

Test at least:

1. Thousands of validations.
2. Hundreds of thousands of snapshot rows.
3. Revalidation history.
4. Many project manifests.
5. Many validators writing during the same interval.

Measure:

1. Validation save latency.
2. Snapshot refresh latency.
3. Report refresh latency.
4. State-store write amplification.
5. API rate-limit behavior.

## Open Decisions

The next review must decide:

1. Confirm real multi-user Bucket permissions for validator read/write access under OAuth.
2. Should project state default to private even if audio is public?
3. Are validators allowed to write directly to state stores with their HF identity?
4. Do we support organization-owned projects in the first implementation?
5. Do we support email-only invites before OAuth/store ownership is redesigned?
6. Is `dataset_token` removed entirely from the default app flow or kept as admin-only legacy compatibility?
7. What export format is the long-term portable archive of project state?

## Recommended Implementation Order

1. Auth hardening.
2. Backend interfaces and contract tests.
3. HF project-state backend spike.
4. Project creation/connect wizard.
5. Migration/export tooling.
6. Documentation and production defaults.

## Not Recommended

Avoid these as default production decisions:

1. Treating the free Space filesystem as durable state.
2. Storing all third-party projects indefinitely in a maintainer-owned free Supabase project.
3. Requiring every non-technical admin to configure a separate database before creating a project.
4. Persisting reusable broad HF tokens in project records unless an explicit secure legacy mode is designed.
5. Writing one HF repository commit for every single validation action.

## Initial Backlog Seed

### Epic A: Production identity

1. Add auth-mode config.
2. Add OAuth session adapter.
3. Disable username-only production login.
4. Add auth regression tests.

### Epic B: Project state contracts

1. Define storage contracts.
2. Move current Supabase persistence behind contracts.
3. Move filesystem persistence behind contracts.
4. Add backend contract tests.

### Epic C: HF-owned project state spike

1. Prototype state store creation/connect.
2. Prototype event append and snapshot update.
3. Prototype ACL read/write.
4. Prototype report load.
5. Document limits and permission behavior.

### Epic D: Product flows

1. Project creation wizard.
2. Project connect wizard.
3. Project export/archive UI.
4. Admin storage visibility/status panel.

### Epic E: Migration and operations

1. Supabase export.
2. HF state import.
3. State verification report.
4. Deployment docs.
5. Privacy and security docs.

## Definition of Done for the Redesign

The redesign is complete when:

1. The production app does not rely on username-only identity.
2. The default public/community architecture does not charge project storage growth to the maintainer.
3. Project durable state survives Space redeploys and app updates under admin-owned storage.
4. The standard project flow does not require stored project HF tokens.
5. ACL, invites, validation writes, and reports are covered by backend contract tests.
6. One migrated real project completes validation workflow without Supabase.
7. Admins can understand where their project state lives and how to export it.

## Implementation Progress

### 2026-05-24: Initial safety foundation

Implemented on branch `feature/project-state-security-privacy-review`:

1. Added destructive-write protection for bootstrap persistence.
   - Project and ACL removals are blocked unless the caller declares an explicit project-delete intent.
   - Supabase persistence no longer broadly deactivates missing projects or ACL rows during ordinary create/update/invite operations.

2. Added local JSON backups for filesystem state.
   - Existing bootstrap JSON files are copied into `.backups/` before replacement.
   - Validation `current.json` snapshots are also backed up before updates.

3. Added append-only snapshot recovery.
   - If filesystem `current.json` is missing or invalid, the current validation snapshot is rebuilt from validation event logs.
   - Malformed event lines are skipped instead of breaking report/validation loading.

4. Added production identity guardrails.
   - New `BIRDNET_AUTH_MODE` config: `auto`, `hf_token`, `username`, or `username_or_token`.
   - In automatic mode, Hugging Face Spaces without demo bootstrap require HF-token identity instead of username-only login.
   - Local/demo mode preserves username login for testing.

5. Added explicit repository contracts for future state backends.
   - Project catalog.
   - Project access.
   - Project invites.
   - Validation events.
   - Current validation snapshots.

Validation:

1. `pytest -q`: 182 passed.
2. `python scripts/check_deployment.py`: passed.

### 2026-05-24: Automatic HF companion state repository on project creation

Implemented on branch `feature/project-state-security-privacy-review`:

1. Added a Hugging Face project state store initializer.
   - New projects derive a private companion dataset repo from the audio dataset repo.
   - Example: `jrrribeiro/upload_test2` creates or connects `jrrribeiro/upload_test2_state`.
   - The state repo is created as a private dataset repository.

2. Added the first state-store manifest files.
   - `README.md`
   - `project.json`
   - `acl.json`
   - `invites.json`
   - `snapshots/current.json`
   - `events/.gitkeep`

3. Added overwrite protection for existing companion state repos.
   - If `project.json` already exists, the app connects the existing state repo without rewriting current state.
   - If the repo has files but no project manifest, initialization is blocked to avoid accidental state loss.

4. Added project-level state metadata.
   - `state_backend`
   - `state_repo_id`
   - `state_schema_version`
   - `state_status`

5. Reduced token persistence risk during project creation.
   - A session/env token may be used to create the private state repo.
   - The app only stores a project dataset token when the admin explicitly enters one.

Validation:

1. `pytest tests\unit\test_hf_project_state_store.py tests\unit\test_app_factory_audio_helpers.py tests\unit\test_admin_panel_manager.py tests\unit\test_supabase_state.py -q`: passed.
2. `python -m compileall app.py src`: passed.

### 2026-05-24: HF project-state validation backend spike

Implemented on branch `feature/project-state-security-privacy-review`:

1. Added an HF-backed validation repository for companion state repos.
   - Writes one append-only event JSON under `events/YYYYMMDD/`.
   - Updates `snapshots/current.json` with the latest state.
   - Preserves optimistic version checks before writing.

2. Added snapshot recovery from events.
   - If `snapshots/current.json` is missing or invalid, current state can be rebuilt from event files.
   - Event listing ignores unrelated project events and non-event placeholders.

3. Added a project-aware validation router.
   - Default behavior remains unchanged.
   - Filesystem/Supabase remain the active path unless `BIRDNET_HF_PROJECT_STATE_WRITES_ENABLED=true`.
   - When enabled, projects with `state_backend=hf_project_store`, a `state_repo_id`, and an available token can route writes/reads to the admin-owned `_state` repo.

4. Added Settings health visibility.
   - The Settings tab now exposes whether HF project-state writes are enabled.

Validation:

1. `pytest tests\unit\test_hf_project_state_validation_repository.py tests\unit\test_project_aware_validation_repository.py tests\unit\test_runtime_config.py tests\unit\test_hf_project_state_store.py -q`: passed.
2. `python -m compileall app.py src`: passed.

### 2026-05-24: HF project-state administrative sync

Implemented on branch `feature/project-state-security-privacy-review`:

1. Added project-owned administrative state sync.
   - `project.json` stores the project manifest.
   - `acl.json` stores project-scoped admins and validators.
   - `invites.json` stores pending invites for the project.

2. Added safety boundaries for admin-state sync.
   - Dataset/project tokens are not serialized into the `_state` repo.
   - Sync is filtered by project and does not include users/invites from other projects.
   - Sync only writes `project.json`, `acl.json`, and `invites.json`.
   - Validation history under `events/` and `snapshots/` is never deleted or rewritten by admin sync.

3. Connected admin workflows to the `_state` repo when HF project-state writes are enabled.
   - Project creation.
   - Project token updates.
   - Project archival/deletion from the workspace.
   - Direct user assignment.
   - Invite creation.
   - Invite revocation.
   - Invite acceptance/rejection.

4. Added safe archive behavior.
   - Deleting a project from the validator workspace does not delete the HF audio dataset or the `_state` repo.
   - The `_state` manifest is marked archived and ACL/invites are cleared while validation history remains intact.

Validation:

1. `pytest tests\unit\test_hf_project_state_store.py tests\unit\test_app_factory_audio_helpers.py tests\unit\test_admin_panel_manager.py tests\unit\test_project_aware_validation_repository.py tests\unit\test_hf_project_state_validation_repository.py -q`: passed.
2. `python -m compileall app.py src`: passed.

### 2026-05-24: HF project-state recovery bootstrap

Implemented on branch `feature/project-state-security-privacy-review`:

1. Added explicit recovery from admin-owned `_state` repositories.
   - New env var: `BIRDNET_HF_PROJECT_STATE_REPOS`.
   - Accepts comma, semicolon, or newline-separated repo IDs.
   - Example: `BIRDNET_HF_PROJECT_STATE_REPOS=jrrribeiro/upload_test2_state`.

2. Added a project-state loader.
   - Reads `project.json`, `acl.json`, and `invites.json`.
   - Reconstructs `Project`, project-scoped user roles, and pending invites.
   - Skips archived projects.
   - Rejects future schema versions instead of guessing.

3. Added bootstrap overlay behavior.
   - HF project-state data overlays only the matching project slug.
   - Existing unrelated projects, access entries, and invites are preserved.
   - Loaded ACL/invites replace stale local/Supabase entries for the same project.

4. Added recovery status visibility.
   - Settings health now reports how many HF project-state repos are configured.
   - Bootstrap warnings report missing tokens or repo-specific load failures.

Validation:

1. `pytest tests\unit\test_hf_project_state_store.py tests\unit\test_app_factory_audio_helpers.py tests\unit\test_runtime_config.py -q`: passed.
2. `python -m compileall app.py src`: passed.

### 2026-05-24: OAuth identity and high-frequency Bucket foundation

Implemented on branch `feature/project-state-security-privacy-review`:

1. Added Hugging Face OAuth session integration.
   - The Space login UI uses `gr.LoginButton` in the hosted environment.
   - An OAuth profile and its short-lived user token create the app session directly.
   - Manual HF-token login remains a fallback; username-only login remains restricted by deployment mode.
   - Personal tokens are held in memory for the session and are not serialized into project state.

2. Added least-privilege OAuth metadata.
   - Requests repository read access for authorized audio datasets.
   - Requests contributed-repository access for state repositories created through the app.
   - Broader shared-write permissions are not silently requested before a permission proof.

3. Added an opt-in Hugging Face Storage Bucket validation backend.
   - New configuration flag: `BIRDNET_HF_BUCKET_VALIDATIONS_ENABLED`.
   - When enabled for creation, a private admin-owned validation Bucket is initialized automatically.
   - Validation events and `snapshots/current.json` are updated through mutable Bucket file operations rather than one Git commit per review.
   - Existing projects and the existing `_state` route remain unchanged unless explicitly enabled.

4. Bound Bucket operations to the acting collaborator.
   - Validation saves, queue snapshot reads, progress reads, and exports can route with the signed-in validator identity.
   - A Bucket-backed write fails instead of falling back to an admin token when the validator has no own HF authorization.

Still required before production enablement:

1. A real two-user permission proof in a Space to confirm the safest Bucket sharing model.
2. A multi-validator concurrency/load test at expected review speed.
3. Onboarding/recovery UI for connecting existing state resources and migrating projects.

### 2026-05-24: Bucket reconciliation and persistent conflict visibility

Implemented on branch `feature/project-state-security-privacy-review`:

1. Strengthened Bucket event recovery.
   - Current snapshots are reconciled against append-only validation events.
   - A snapshot overwritten by a parallel writer no longer silently omits valid events for other detections.
   - Event payloads are downloaded through a batched read path rather than one request per event file.

2. Persisted evidence of parallel same-version decisions.
   - If two writers produce different decisions for the same detection version, the reconstructed state is marked with `conflict=true`.
   - The validation queue surfaces that persisted conflict using the existing conflict workflow.
   - CSV/XLSX exports now include `validation_conflict` and `validation_conflict_reason`.

### 2026-05-24: Bounded active event window and compact audit archives

Implemented on branch `feature/project-state-security-privacy-review`:

1. Added automatic Bucket event compaction.
   - An active window of up to 250 individual validation events is retained for reconciliation during ordinary writes.
   - Before the next write after the active window fills, events are rolled into an append-only JSONL audit archive and the reconciled snapshot is updated.
   - The original individual event objects are removed only in the same Bucket batch that writes their archive and snapshot.

2. Kept historical recovery and auditing intact.
   - A missing or corrupt snapshot can be rebuilt from archived and active events.
   - Duplicate event IDs produced by parallel compactors are de-duplicated during reads.
   - If an active event cannot be read safely, compaction aborts instead of deleting it.

3. Added bounded recent-activity reads for the Progress dashboard.
   - The dashboard asks for only the events required for the visible recent-activity page plus one look-ahead row.
   - Bucket-backed reads inspect the active window and only as many newest archive files as necessary.
   - Full CSV/XLSX scientific exports remain based on the complete current detection snapshot.

Remaining validation before real-data enablement:

1. Exercise simultaneous validators against the compacted strategy and measure latency at expected review speed.
2. Confirm multi-user Bucket authorization in a real Space.
3. Implement migration/import tooling for existing non-HF project state resources.

### 2026-05-24: Secure existing-state connection flow

Implemented on branch `feature/project-state-security-privacy-review`:

1. Added an Admin recovery action for existing private `_state` repositories.
   - The admin enters an existing companion repository id in **Project management** and chooses **Connect Existing State**.
   - The app loads its saved project manifest, ACL, and pending invites using the signed-in user's Hugging Face session authorization.
   - Connected project state is made available in the current workspace without rewriting the authoritative `_state` files.

2. Enforced recovery authorization before state is attached to the workspace.
   - The authenticated identity must be listed as `admin` in the stored ACL.
   - Private projects can only be connected by their recorded owner.
   - Missing authenticated tokens and archived state repositories are rejected.

3. Prevented state takeover through the creation screen.
   - If project creation discovers an existing `_state` manifest, it no longer registers the newly entered project definition.
   - The admin is directed to the connection flow so the existing manifest and ACL remain authoritative.

4. Prevented validation-state fragmentation during recovery.
   - A project whose manifest declares Bucket-backed validation state now fails clearly if the deployment has the Bucket backend disabled or the stored Bucket reference is incomplete.
   - It never silently writes recovered validations to filesystem or Supabase instead of the project-owned Bucket.

Still required before production enablement:

1. Exercise simultaneous validators against the compacted strategy and measure latency at expected review speed.
2. Confirm multi-user Bucket authorization in a real Space.
3. Add assisted migration/import verification for legacy filesystem or Supabase project state.

