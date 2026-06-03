# BYO Supabase With Edge Function And Hugging Face Identity Plan

Status: deferred target architecture for public distribution
Created: 2026-05-26
Implementation target: a future branch after the trusted-team HF delivery is stable

## Decision Summary

For a distributed public application, each project administrator should own a
Supabase backend in their own account. The BirdNET Validator app should:

1. authenticate scientists through Hugging Face;
2. let administrators connect a Supabase account through Supabase OAuth;
3. provision or connect an administrator-owned Supabase project;
4. install the validation schema and a project write API in an Edge Function;
5. issue short-lived validation authorization tokens after checking the user's
   Hugging Face identity and project role; and
6. never distribute an administrator database secret to validators.

This model keeps relational-database performance while shifting storage quota,
state ownership, exports, and eventual billing responsibility to the project
administrator.

## Why This Is The Distribution Target

The application must support:

1. thousands or millions of source detections;
2. multiple simultaneous validators;
3. validators capable of submitting one decision every few seconds;
4. complete audit history and reconstructable current state;
5. fast progress dashboards and scientific exports;
6. private pre-publication research data; and
7. projects owned by independent administrators, not by the Space maintainer.

A Supabase database is structurally appropriate for concurrent events,
transactions, queries, and reports. Google Drive is useful for backups but
would require a custom append-only synchronization engine as the primary
backend. Hugging Face organization storage is appropriate for a small trusted
team but requires broad organization write access for validators on the
currently proven free-tier path.

## Verified Constraints And Required Proofs

### Confirmed Through Official Documentation

1. Supabase exposes a Management API intended for platform integrations.
2. Supabase integrations can use OAuth rather than requiring administrators to
   paste personal access tokens.
3. OAuth scopes include project, database, secret, and Edge Function
   management permissions.
4. Supabase Edge Functions can run trusted server code close to the
   administrator-owned database.
5. Hugging Face identity is not listed as a built-in Supabase third-party auth
   provider.

### Must Be Proven Before Product Implementation

1. A regular free Supabase administrator can authorize our OAuth integration.
2. The granted scopes allow creating or selecting a project without manual
   administrator secrets.
3. The integration can install schema migrations in that project, or there is
   an acceptable one-time SQL fallback.
4. The integration can deploy/update an Edge Function in a free project.
5. A validator authenticated only with Hugging Face can submit through that
   function using a short-lived app authorization token.
6. Revocation, replay protection, simultaneous writes, exports, and restore
   paths work under real use.

No production redesign begins until this spike has passed with two distinct
Hugging Face users and a Supabase account owned by the test administrator.

## Ownership Model

### Central Space Owns

The shared Space owns only application runtime concerns:

1. UI and validation workflows;
2. Hugging Face login/session handling;
3. Supabase OAuth integration identity;
4. the private signing key used to issue short-lived validation assertions;
5. minimal encrypted connection registry, only if required to reconnect an
   administrator's backend; and
6. diagnostics and compatibility migrations.

### Administrator-Owned Supabase Project Owns

One Supabase project may contain several BirdNET projects belonging to the
same administrator:

```text
admin Supabase account
  birdnet-validator-state project
    birdnet_projects
    project_members
    project_invites
    detection_catalog
    validation_events
    current_validations
    export_jobs
    audit_events
    edge function: submit-validation
```

It stores project metadata, authorization, validation history, derived current
state, and exports. It does not need to store audio files.

### Hugging Face Dataset Owns

The administrator's Hugging Face dataset continues to own:

1. audio segments;
2. uploader-generated indexes;
3. original BirdNET detection metadata; and
4. stable identifiers required to join exports.

For private audio, validators still need either individual read permission in
Hugging Face or a separately designed read proxy. BYO Supabase solves state
ownership, not private source-audio authorization.

## Authentication And Authorization Model

### Administrator Setup

1. Administrator logs into BirdNET Validator with Hugging Face.
2. Administrator chooses **Connect Supabase Storage**.
3. App redirects to Supabase OAuth consent.
4. Administrator grants only required scopes.
5. App creates or selects a Supabase project owned by that administrator.
6. App provisions schema and Edge Function.
7. App records a backend reference for the BirdNET project.

### Validator Session

1. Validator logs in with Hugging Face only.
2. App verifies identity from the HF session/token.
3. App reads project membership from the administrator-owned backend through a
   read path or cached signed project mapping.
4. When the validator submits a decision, the app issues a signed, short-lived
   authorization assertion containing:

```json
{
  "iss": "birdnet-validator-space",
  "sub": "hf:validator_username",
  "project_id": "uuid",
  "role": "validator",
  "action": "validation:submit",
  "jti": "unique-id",
  "iat": 0,
  "exp": 0
}
```

5. The Supabase Edge Function validates signature, expiry, action, project
   membership, and replay rules before performing the database transaction.

### Trust Boundary

The Space remains the identity broker: it asserts that a Hugging Face user is
currently signed in and permitted to take an action. It does not own the
scientific validation database or distribute database administrator keys.

## Supabase OAuth And Secret Policy

Required principles:

1. Prefer Supabase OAuth over personal access tokens.
2. Request the minimum scopes needed for the provisioning phase.
3. Keep OAuth refresh tokens encrypted at rest if persistent reconnection is
   required.
4. Separate provisioning authority from runtime validation authority.
5. Do not save `service_role` or modern `secret` API keys in project rows.
6. Never send privileged Supabase credentials to a browser or validator.
7. Provide a **Disconnect Supabase** action that revokes local references and
   explains how to revoke authorization in Supabase.

Candidate scopes must be finalized after the spike. Expected needs include:

| Capability | Expected purpose |
|---|---|
| Project read/write | Select or create admin-owned Supabase project |
| Database write | Install/update schema if available to the integration |
| Edge Function write | Deploy validation submission handler |
| Secret write | Install function verification material without exposing it |

## Database Schema

### `birdnet_projects`

| Column | Purpose |
|---|---|
| `project_id` | Stable UUID |
| `project_slug` | User-facing key |
| `owner_hf_username` | Project administrator identity |
| `dataset_repo_id` | HF source dataset |
| `created_at`, `updated_at` | Audit timestamps |
| `active` | Archive control |

### `project_members`

| Column | Purpose |
|---|---|
| `project_id` | Project scope |
| `hf_username` | Verified collaborator identity |
| `role` | `admin`, `validator`, or `viewer` |
| `active` | Revocation control |
| `granted_by`, `granted_at` | Audit |

### `validation_events`

Append-only source of truth:

| Column | Purpose |
|---|---|
| `event_id` | UUID/idempotency key |
| `project_id` | Scope |
| `detection_key` | Stable segment identifier |
| `actor_hf_username` | Validator |
| `status`, `corrected_species`, `notes` | Scientific review |
| `expected_version`, `new_version` | Concurrency control |
| `created_at` | Audit ordering |
| `auth_jti` | Replay/audit link |

### `current_validations`

Fast derived state keyed by `(project_id, detection_key)`. It must be updated
within the same database transaction as event insertion.

### Detection Catalog Policy

The full source detection table may be loaded from HF on demand or cached in
Supabase when necessary for reports. The original dataset index remains the
authoritative source for audio, species, confidence, start time, and end time.

## Edge Function Write Contract

Suggested endpoint:

```http
POST /functions/v1/submit-validation
Authorization: Bearer <short-lived app assertion>
Content-Type: application/json
Idempotency-Key: <event uuid>
```

Request:

```json
{
  "project_id": "uuid",
  "detection_key": "stable-key",
  "status": "positive",
  "corrected_species": null,
  "notes": "",
  "expected_version": 0
}
```

Function requirements:

1. verify signature against installed public key or shared verifier secret;
2. reject expired or replayed assertions;
3. verify active member role;
4. verify request project equals asserted project;
5. execute atomic event insert and snapshot update;
6. return the new version or a structured conflict;
7. append an audit record for authorization failures and accepted writes.

## Conflict And Throughput Strategy

1. Every decision is an immutable event.
2. Current state updates use an atomic database transaction or SQL RPC.
3. Two decisions for the same expected version return a conflict for review;
   neither is silently discarded.
4. The UI may batch network calls only if every event keeps an idempotency key.
5. Dashboard queries read `current_validations`, not all historical events.

The expected workload of rapid human validation is natural for PostgreSQL and
must be tested with simulated multi-user bursts.

## Administrator Experience

New UI surfaces:

1. **Storage Setup**: connect Supabase, choose existing/create backend, verify
   schema and function health.
2. **Project Management**: create BirdNET project in connected backend.
3. **Member Management**: app-level invitation and role controls persisted in
   the administrator's Supabase.
4. **Health**: backend owner, connection state, migration version, last
   successful write, last export, and reconnect action.
5. **Export And Backup**: prepare CSV/XLSX and optionally back them up to
   administrator-owned Google Drive or HF resources.

## Implementation Phases

### Phase 0 - Feasibility Spike

1. Register a private Supabase OAuth integration.
2. Complete OAuth from a local test app and from an HF Space.
3. List/create a free Supabase project through Management API.
4. Attempt automated schema provisioning.
5. Attempt Edge Function deployment and secret installation.
6. Submit one validation using a separately authenticated HF validator.
7. Revoke OAuth and confirm configuration writes stop.

Exit gate: all required operations succeed without a manually supplied
administrator database key, or the remaining one-time manual step is explicitly
accepted.

### Phase 1 - Backend Contracts

1. Add `supabase_owned` project backend type.
2. Add encrypted connection-registry interface.
3. Implement provisioning service behind a mockable client.
4. Implement Edge Function source and schema migrations.
5. Add connection health and migration version checks.

### Phase 2 - Identity Broker

1. Require verified HF login in production.
2. Implement short-lived assertion signer and key rotation.
3. Deploy verifier configuration to administrator Edge Functions.
4. Add membership checks, expiration, idempotency, and replay defense.

### Phase 3 - Product Workflows

1. Implement **Connect Supabase Storage** UI.
2. Create projects and membership records in admin-owned storage.
3. Route validation, progress, and export operations through that backend.
4. Remove shared project token persistence from the default flow.

### Phase 4 - Reliability And Migration

1. Import projects/events from central Supabase or trusted-team HF Buckets.
2. Add automated export backups.
3. Load-test simultaneous validation.
4. Security review before enabling for third-party projects.

## Test Matrix

| Area | Required tests |
|---|---|
| OAuth setup | consent, revoked token, expired token, denied scopes |
| Provisioning | new project, existing project, migration retry, failed setup recovery |
| Identity | valid HF actor, removed member, forged assertion, expired assertion |
| Writes | idempotent retry, same-audio conflict, parallel different audio submissions |
| Reports | ongoing project export, completed export, species aggregation |
| Privacy | no secret in UI/log/export, unauthorized user cannot query or submit |
| Resilience | Space rebuild, Edge Function redeploy, Supabase outage and recovery |

## Acceptance Criteria

The distribution architecture is acceptable only when:

1. project state is durably stored in the administrator's Supabase account;
2. validators require only their Hugging Face identity during routine work;
3. no validator receives database administrator secrets;
4. concurrent validation writes do not silently overwrite each other;
5. the administrator can revoke a validator and immediately stop new writes;
6. a Space rebuild does not erase projects or validation history;
7. scientific CSV/XLSX exports remain complete and reproducible; and
8. the app exposes the ownership, quota, and recovery status clearly.

## Official References

1. Supabase Management API: <https://supabase.com/docs/reference/api/introduction>
2. Supabase OAuth integrations:
   <https://supabase.com/docs/guides/integrations/build-a-supabase-oauth-integration>
3. Supabase for Platforms:
   <https://supabase.com/docs/guides/integrations/supabase-for-platforms>
4. Supabase Edge Functions:
   <https://supabase.com/docs/guides/functions>
5. Supabase Row Level Security:
   <https://supabase.com/docs/guides/database/postgres/row-level-security>
6. Supabase third-party auth:
   <https://supabase.com/docs/guides/auth/third-party/overview>
7. Hugging Face OAuth:
   <https://huggingface.co/docs/hub/oauth>
