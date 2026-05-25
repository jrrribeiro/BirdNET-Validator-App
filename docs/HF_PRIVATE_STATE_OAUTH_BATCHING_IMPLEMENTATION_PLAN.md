# Private HF State + OAuth + Batching Implementation Plan

Status: permission proof failed; retained as investigation record
Created: 2026-05-25
Target branch: `feature/project-state-security-privacy-review`

## Decision Summary

Adopt a private Hugging Face companion dataset repository (`*_state`) as the
project-owned durable state resource, authenticated through each user's Hugging
Face OAuth session and optimized for frequent validation through batched,
append-only writes.

This strategy is intended for both collaborative and private projects, with one
important boundary:

- The private `_state` repository protects project state, ACL, invites, and
  validation decisions.
- Access to audio files is independent. Validators of a private audio dataset
  must also have Hugging Face read access to that dataset.
- An app invitation is an application-level authorization rule; it does not by
  itself grant Hub permissions to an existing private audio dataset.

The experimental admin-owned private Bucket path must remain disabled by
default. The two-user test demonstrated that an invited validator could not
access the personal private Bucket using their own Hugging Face authorization.

The private personal `_state` OAuth path also cannot become the default
validation backend as designed. On 2026-05-25, an OAuth-authenticated admin
ran the permission diagnostic against an initialized private state repository.
The manifest was read, but a direct diagnostic commit was rejected with `403
Forbidden` and the Hub required `create_pr=1`. Pull Request-only contributions
are appropriate for reviewed content submissions, not a high-frequency
validation event stream.

## What Is Confirmed And What Requires Proof

### Confirmed by official documentation

1. Hugging Face OAuth supports `contribute-repos`, which can create repositories
   and access repositories created by the application, without granting access
   to other personal repositories.
2. Free users and free organizations include 100 GB of private Hub storage.
   Storage limits apply to datasets, repositories, and Buckets.
3. Git-backed Hub repositories are not intended for database-style high-write
   workloads; the Hub experience can degrade after thousands of commits.
4. Hub action limits include repository commits and are not documented as fixed
   public numbers. The general free-user API request allowance is measured in
   five-minute windows and may change.

### Permission proof result

The following proof was designed before implementation:

1. Admin A signs into the Space through OAuth.
2. Admin A creates a private `*_state` repository through the app.
3. Validator B signs into the same Space through OAuth.
4. Validator B accepts an app invitation.
5. Validator B reads the private state manifest and attempts one test
   validation write using only B's OAuth token.
6. Admin A reloads the project and observes that write.

Observed result:

1. A manual-token `upload_test5_state` attempt was invalid for testing OAuth
   scopes and was discarded as evidence.
2. Admin A then authenticated through OAuth and executed the diagnostic on
   `jrrribeiro/upload_test6_state`.
3. The diagnostic read the initialized repository but could not commit its
   proof file directly to `main`.
4. The Hub returned `403 Forbidden` and explicitly required `create_pr=1`.

Conclusion: this strategy failed before testing validator B. If even the
creating OAuth identity cannot perform direct state writes, `contribute-repos`
cannot support durable batched validation updates without a Pull Request merge
workflow or an administrator credential. Neither meets the target workflow.

## Target Ownership And Privacy Model

### State repository

For every project, the app creates:

```text
owner-or-org/audio_dataset_state
```

The repository is private regardless of whether the audio dataset is public,
gated, or private. It is owned by the project admin's Hugging Face namespace,
or by a project organization explicitly chosen by the admin. Its storage usage
belongs to that owner, not to the app maintainer.

### Project categories

| Project type | Audio dataset access | `_state` visibility | Validator requirement |
|---|---|---|---|
| Public collaborative | Public resolver access or authenticated read | Private | OAuth access to `_state` after permission proof |
| Gated collaborative | Approved gated access | Private | Own HF account approved for audio plus `_state` authorization |
| Private collaborative | Private repo read granted by owner/org | Private | Own HF account authorized for both audio and `_state` |
| Private individual | Admin-only private audio access | Private | Admin OAuth only |

The application may present invitations and roles, but must never suggest that
an invitation alone unlocks a private audio repository.

## Security Principles

1. No permanent administrator HF token is persisted in project records, state
   files, exports, logs, or browser-facing values.
2. Every write is attributed to the signed-in validator identity.
3. Collaborative state writes require that user's OAuth token; they never
   silently fall back to a stored administrator token.
4. `acl.json` governs application roles only after Hub authentication succeeds.
5. Private state recovery requires authenticated admin access and a recorded
   admin role in `acl.json`.
6. Events are append-only; snapshots are derived and recoverable.
7. Pending, locally accepted decisions must be visible to the user until
   durable synchronization is acknowledged.

## Target State Layout

Keep low-frequency control files small and human-inspectable:

```text
project.json
acl.json
invites.json
checkpoints/current.json
events/
  batches/
    <validator_username>/
      YYYYMMDD/
        <batch_id>.jsonl
manifests/
  batches.jsonl              # optional compact audit manifest
conflicts/
  <detection_key>.json       # optional materialized unresolved conflicts
```

### Control-plane data

`project.json`, `acl.json`, and `invites.json` are written only for project
administration operations. They must contain no token.

### Validation event batch

Each accepted decision is represented as an immutable event. A JSONL batch
contains multiple events from one authenticated validator:

```json
{
  "schema_version": 2,
  "event_id": "uuid",
  "batch_id": "uuid",
  "project_slug": "upload_test5",
  "detection_key": "stable-key",
  "status": "positive",
  "corrected_species": null,
  "notes": "",
  "validator": "validator_username",
  "client_sequence": 17,
  "expected_version": 0,
  "timestamp": "2026-05-25T12:00:00Z"
}
```

The server must derive validator identity from the authenticated session. It
must not trust a client-provided username.

## Batching And Durability Strategy

The existing `_state` repository implementation creates one Hub commit per
validation. Replace that behavior before real high-volume use.

### Outbox first

Batching creates a short interval before remote durability. A validation must
not disappear if the Space restarts during that interval.

Implement a per-user, per-project outbox:

1. Before advancing to the next audio, store the pending event in browser
   durable storage (IndexedDB preferred; local storage only for a small
   prototype).
2. Render the decision as `pending sync` in the interface.
3. Flush outbox events in batches to the private `_state` repository.
4. Remove local entries only after the remote batch commit is confirmed.
5. On browser reload or reconnect, restore and retry unacknowledged events.
6. Provide a visible `Synced`, `Syncing`, `Offline/Pending`, or `Conflict`
   status; never silently claim durability.

### Initial batching policy

Start conservatively and measure:

| Trigger | Initial value |
|---|---:|
| Batch size | 25 validation events |
| Flush interval while pending | 15 seconds |
| Immediate flush | project change, sign out, export request, manual Sync |
| Retry | exponential backoff for transient HTTP/429/5xx failures |
| Maximum local pending events | 1,000 before validation is paused for sync |

At one decision every 3 seconds, one active validator would produce about 20
events/minute. The initial policy should ordinarily reduce approximately 20
commits/minute to roughly 1 commit/minute per active validator, before further
measurement.

### Snapshot policy

Do not overwrite a shared full snapshot on every batch:

1. Validation batches are the durable source of truth.
2. Rebuild or incrementally reconcile current state from batch files.
3. Write `checkpoints/current.json` only periodically, such as every 20
   confirmed batches or on admin export/maintenance.
4. If parallel writers validate the same detection from the same base version,
   surface a conflict rather than discarding either event.

This prevents concurrent validators from overwriting one another's recent
decisions through frequent shared snapshot rewrites.

## Implementation Phases

## Phase 0 - Permission Proof And Architecture Gate

Objective: establish whether `contribute-repos` supports the required
two-account private `_state` collaboration.

Implementation status (2026-05-25): completed, failed permission gate.

Changes:

1. Disable Bucket creation for newly created test projects. Implemented.
2. Add a clearly labeled `State authorization test` action visible to project
   admins and invited validators in a test project.
   Implemented as **Private state authorization** in the **Projects** tab.
3. Record diagnostics without tokens: acting username, repository id,
   read/write result, HTTP category, and timestamp.
   Partially implemented: successful read/write proof writes actor, repo,
   project, timestamp, and diagnostic id; failed attempts are returned to the
   user without storing tokens.
4. Keep the test write isolated under `diagnostics/oauth-permission-proof/`.
   Implemented.
5. Reject state validation submissions that would fall back from a signed-in
   collaborator to an administrator token. Implemented.
6. Distinguish OAuth sessions from manual-token sessions and reject manual
   token creation/proof attempts in the hosted Space. Implemented after the
   first attempted test used manual tokens and therefore did not exercise
   `contribute-repos`.

Test note:

- `upload_test5_state` was created and tested through manual-token sessions;
  its failed collaborator write is not a valid OAuth permission proof.
- The next proof must create a new repository, such as `upload_test6_state`,
  after the admin completes both OAuth login steps in the hosted Space.

Tests:

1. Unit tests ensuring only the session OAuth token is passed to the state
   repository.
2. Unit test rejecting admin-token fallback for collaborator state writes.
3. Manual Space test with two free HF accounts and a fresh private `_state`
   repo.

Exit gate outcome:

- Failed. Do not proceed to the `_state` batching phases as a production
  backend.
- Preserve the code only as a permission diagnostic until the next backend
  strategy is chosen.
- Do not mask the failure using an administrator token.

## Phase 1 - Backend Selection And Removal Of Unsafe Fallbacks

Objective: make private `_state` the explicit validation backend for new
projects after the proof passes.

Changes:

1. Add a backend constant such as `hf_private_state_batched`.
2. New projects default to private `_state` validation storage; Bucket is shown
   as experimental/disabled and is not selected automatically.
3. Separate token policies:
   - audio read token: acting user's OAuth token;
   - state read/write token: acting user's OAuth token;
   - project creation token: creating admin's OAuth token.
4. Remove project `dataset_token` from the normal collaborative creation flow.
5. Reject collaborative state writes without an authenticated OAuth token.
6. Preserve old Bucket/Supabase projects as migration sources only.

Tests:

1. Router tests for each backend declaration.
2. Tests proving no project token or environment token is used for a
   collaborator write.
3. Tests for public, gated, and inaccessible private audio messages.

## Phase 2 - Batched Event Repository

Objective: replace one-commit-per-validation behavior.

Changes:

1. Implement `HfBatchedProjectStateValidationRepository`.
2. Add immutable JSONL batch commits partitioned by validator and date.
3. Add batch idempotency by `event_id` and `batch_id`.
4. Add event reconciliation, version conflict detection, and a bounded
   checkpoint writer.
5. Replace full-repository scans with checkpoint plus new-batch discovery.

Tests:

1. Serialize/deserialize event batches.
2. Retry the same batch without duplicate validations.
3. Merge batches from two validators.
4. Detect two incompatible decisions for one base version.
5. Recover a snapshot from batches after deleting/corrupting a checkpoint.
6. Assert commit count is proportional to batches, not decisions.

## Phase 3 - Durable Client Outbox And UX

Objective: maintain user trust during batching and transient network failures.

Changes:

1. Implement browser-side persistent outbox.
2. Add session handoff for flush/retry endpoints.
3. Show sync state near the validation action area and in Progress.
4. Add `Sync now` and prevent sign-out without an explicit pending-sync
   warning.
5. Exports must either flush first or identify unconfirmed local decisions
   clearly.

Tests:

1. Simulate failed network response and browser reload.
2. Confirm restored pending decision is resent once.
3. Confirm synced decision is removed from local storage.
4. Confirm project switching does not mix pending events.

## Phase 4 - Project Privacy And Access UX

Objective: make private/public access behavior understandable and safe.

Changes:

1. During project creation, distinguish:
   - audio visibility/access policy;
   - private validation-state policy.
2. Before validation starts, run an audio-read check and state-write
   authorization check for the signed-in user.
3. In Admin, show collaborator readiness statuses:
   - invited in app;
   - state access verified;
   - audio access verified;
   - ready to validate.
4. For private/gated audio, show instructions to grant Hub access externally;
   do not imply the app invitation performs it.

Tests:

1. Public audio/private state workflow.
2. Private audio with authorized validator.
3. Private audio without authorization.
4. Authorized state but inaccessible audio and the inverse.

## Phase 5 - Migration And Recovery

Objective: protect existing test/real work during transition.

Changes:

1. Classify current projects by backend: filesystem, Supabase, `_state`
   single-event commits, or Bucket.
2. Provide admin-only migration preview with counts and target state repo.
3. Import prior events into batch files without deleting the source.
4. Verify counts, latest statuses, versions, conflicts, and export output.
5. Switch backend only after verification succeeds.
6. Retain source as read-only recovery material until the admin confirms.

Tests:

1. Migration from existing `_state` event layout.
2. Migration from Supabase export fixture.
3. Migration abort leaves the original backend active.
4. CSV/XLSX equality before and after migration for confirmed state.

## Phase 6 - Scale And Multi-User Acceptance

Objective: qualify the backend before real scientific validation.

Test scenarios:

1. Two free HF accounts, one admin and one validator.
2. Public audio dataset with private `_state`.
3. Private audio dataset where both accounts have Hub access.
4. At least two concurrent validators operating on the same species queue.
5. Simulated validation rate of one decision every 3 seconds per validator.
6. Network failure, retry, browser refresh, and Space restart during pending
   sync.
7. Export while validation is ongoing.

Acceptance criteria:

1. No administrator token stored or transmitted to a validator.
2. No acknowledged validation lost after refresh or restart.
3. Duplicate retries do not duplicate decisions.
4. Conflicts are visible and exportable.
5. The number of commits stays near configured batch count, not click count.
6. The admin can recover a project solely from its private `_state` repo and
   authenticated HF access.
7. The final CSV/XLSX contains the complete detection table and all confirmed
   validation fields.

## Fallback Decision If OAuth Permission Proof Fails

If validator B cannot write an admin-created private `_state` repository with
`contribute-repos`, do not restore stored admin tokens as the default model.

Evaluate in this order:

1. A free Hugging Face organization owned by the project team, with the audio
   and `_state` resources owned by that organization and Hub permissions
   explicitly managed there. This keeps quota with the project owner but may
   expose broader organization write access than desired on the free tier.
2. Per-admin external database configuration, such as each project admin
   connecting their own Supabase project. This preserves cost ownership but
   introduces setup burden and free-tier database constraints.
3. A paid or institution-supported organization architecture only as an
   optional advanced deployment, never as the default requirement for users.

## Implementation Order

Do not implement batching first. The correct order is:

1. Phase 0 permission proof.
2. Phase 1 safe routing and removal of unsafe collaborative fallback.
3. Phase 2 batched remote event storage.
4. Phase 3 durable outbox.
5. Phase 4 access/readiness UX.
6. Phase 5 migration.
7. Phase 6 qualification.

This sequence avoids optimizing a storage route that may not satisfy the core
multi-account permission requirement.

## Official References

- Hugging Face OAuth scopes: <https://huggingface.co/docs/hub/oauth>
- Hugging Face storage limits: <https://huggingface.co/docs/hub/en/storage-limits>
- Hugging Face rate limits: <https://huggingface.co/docs/hub/en/rate-limits>
- Hugging Face organization security: <https://huggingface.co/docs/hub/en/organizations-security>
