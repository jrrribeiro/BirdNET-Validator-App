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
| Hugging Face project store | recommended community/default mode |

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

1. Is the first HF state backend a Storage Bucket or companion dataset?
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

