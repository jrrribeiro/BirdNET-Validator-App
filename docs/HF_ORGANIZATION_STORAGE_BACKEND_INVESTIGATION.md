# Hugging Face Organization-Owned State Backend Investigation

Status: investigation and permission-spike plan; not approved for production
Created: 2026-05-25
Target branch: `feature/project-state-security-privacy-review`

## Purpose

Evaluate whether a free Hugging Face organization owned and managed by each
project team can provide persistent, collaborative validation state without:

1. storing an administrator token in the app;
2. making the application maintainer pay for user project state;
3. writing one Git commit per validation; or
4. requiring a paid Hugging Face plan for ordinary projects.

This investigation follows two failed personal-resource experiments:

1. A private Bucket owned by one user's personal namespace was not accessible
   for validation by an invited second user using their own authorization.
2. A private personal `_state` repository could be read through OAuth, but a
   direct state commit was rejected and required a Pull Request. That is not a
   suitable write path for rapid validation work.

## Initial Decision

The organization candidate must be tested with an **organization-owned private
Storage Bucket** for validation events and snapshots, not a Git-backed dataset
repository as its frequent-write store.

Reason: Hugging Face documents Buckets as mutable object storage for changing
files, logs, and intermediate state, without Git history or Pull Requests.
Repositories remain useful only for infrequent metadata, export checkpoints, or
portable backups.

Proposed split, only if the permission test succeeds:

| Content | Candidate storage | Expected frequency |
|---|---|---|
| Audio and original detection index | Existing dataset, public or organization-shared if private | Read-only during validation |
| Project manifest and ACL policy | Small private organization repository or Bucket metadata files | Infrequent |
| Validation events and live snapshot | Private organization-owned Bucket | Frequent/batched writes |
| Scientific CSV/XLSX export checkpoint | Download on demand, optionally archived in project storage | Infrequent |

## Confirmed From Official Hugging Face Documentation

### Cost And Quota

1. A free user **or free organization** currently includes `100GB` of private
   Hub storage.
2. Storage limits apply to repositories and Buckets.
3. Storage above free allowances or upgraded organization features may incur
   charges; a project must never be silently upgraded or attached to the
   app maintainer's billing.

Practical consequence: a state-only Bucket is expected to remain small. A
private audio dataset also placed inside the organization consumes the same
private allowance; multiple large private audio projects can exceed `100GB`.
Public audio does not need to be duplicated into private validation state.

### Write Performance Model

1. Storage Buckets are available to users and organizations.
2. Buckets are mutable, non-versioned storage and do not use Git commits or
   Pull Requests.
3. The Python client supports batch Bucket writes.
4. Git-backed repositories are not intended to behave as frequently written
   databases; the Hub notes degraded experience after thousands of commits.

Practical consequence: organization-owned Buckets are the only HF-native
candidate here that matches hundreds of fast decisions without generating a
commit stream.

### Organization Access Control

Free organizations support member roles:

| Role | Relevant ability |
|---|---|
| `read` | Read organization repositories and metadata only |
| `contributor` | Write only repositories created by that member |
| `write` | Write, create, delete, or rename resources across the organization |
| `admin` | Administrative control |

Fine-grained Resource Groups are documented as part of **Team and Enterprise**
plans, not the free organization model.

Practical consequence: a free organization cannot yet be assumed to grant a
validator write access to only one project's private state resource. A
validator given organization-level write power may be able to modify other
resources in that same organization.

### OAuth Scope Gap

The current app requests:

```yaml
hf_oauth_scopes:
  - read-repos
  - contribute-repos
```

Hugging Face documents that OAuth users may authorize selected organization
access. However, documentation does not establish that the current app scopes
permit direct write access to an organization-owned private Bucket for an
organization member.

Practical consequence: no OAuth scope change should be deployed as a product
feature until a small organization Bucket permission spike succeeds.

## Security Assessment

### What This Model Would Solve

If proven, it would provide:

1. State owned by the project team organization, not by this app maintainer.
2. Each validator authenticating with their own Hugging Face identity.
3. Mutable state writes without Git commit limits.
4. Persistence across Space deployments and restarts.

### What It Does Not Solve On The Free Tier

It does not offer strong project-level isolation inside one shared organization.
Application ACL checks prevent normal UI misuse, but cannot prevent a member
with Hub-level write permission from modifying Bucket content outside the app.

Therefore:

1. A single free organization containing unrelated projects is not acceptable
   when validators must be isolated from each other's project data.
2. A team organization is acceptable only where all users receiving write
   access are trusted at the organization level.
3. Creating one free organization per project might isolate projects, but it
   must not be assumed to be a scalable or supported product design without
   explicit validation of Hugging Face policy and operational burden.
4. Paid Resource Groups would provide a stronger permission boundary, but
   cannot be a requirement for the free default workflow.

## Private And Public Dataset Scenarios

| Audio dataset | Organization state Bucket | Validator requirement | Candidate result |
|---|---|---|---|
| Public personal dataset | Private organization Bucket | Write access to state Bucket only | Best free candidate to test |
| Private organization dataset | Private organization Bucket | Read audio and write state | Testable, but audio counts toward private quota |
| Private personal admin dataset | Private organization Bucket | Personal dataset sharing still unresolved | Not a complete collaborative solution |

The first permission spike should use a public dataset and tiny fake state so
that it tests state authorization only, not private audio sharing.

## Permission Spike: No App Changes Yet

### Goal

Prove whether two free accounts can read and write a private
organization-owned Bucket with their own Hugging Face credentials, and
document the minimum organization role required.

### Setup

1. Create one temporary free Hugging Face organization for the spike.
2. The admin account creates a private Bucket inside that organization, for
   example `<test-org>/validator-state-permission-test`.
3. Do not upload real recordings or validation data; use two small JSON files.
4. Invite the second account into the organization first as `contributor`.

### Credential Test A: User Tokens Outside The App

This isolates Hugging Face permission behavior before changing OAuth scopes.

1. Admin writes `diagnostics/admin-write.json` to the Bucket using the admin's
   own temporary token.
2. Validator attempts to read that file using the validator's own token.
3. Validator attempts to write `diagnostics/validator-contributor-write.json`
   using their own token while assigned `contributor`.
4. If the write fails, change only the test member role to `write`.
5. Validator attempts `diagnostics/validator-write-role-write.json`.
6. Admin reads the validator file using the admin token.

Expected likely result based on the role documentation: `contributor` is
insufficient because the validator did not create the admin-created Bucket;
`write` may succeed but grants organization-wide write powers.

### Credential Test B: OAuth In The Space

Run only if Test A proves an acceptable organization permission model.

1. Add a narrowly scoped experimental OAuth/Bucket diagnostic in this feature
   branch only.
2. Request only the organization-capable scope proven necessary by an
   explicit test; never store returned OAuth tokens in project state.
3. Admin signs in through OAuth and writes one diagnostic object.
4. Validator signs in through OAuth and writes a second diagnostic object.
5. Admin confirms both files and the actor identity recorded in each object.

Pass condition: both authenticated users can directly write private state using
their own authorization, without PRs or shared tokens.

Fail condition: either account cannot write, writes require an admin token, or
OAuth forces broader access than accepted by the security policy.

### Throughput Test C: Batch Validation Writes

Run only if both permission tests pass.

1. Generate simulated validation events with no audio data.
2. Test batches of `25`, `100`, and `250` decisions per Bucket write.
3. Measure write latency, read/reconciliation latency, conflicts, and rate
   limit responses with two simultaneous users.
4. Confirm export reconstruction and recovery after application restart.

Pass condition: validation remains responsive at the expected working rate
(approximately one decision every three seconds per active validator) without
Git commits and without data loss.

## Gate For Implementation

Do not implement this backend as the app default unless all items are true:

1. An organization-owned Bucket can be written by both test accounts using
   only their own authorization.
2. The required free-tier role and its security implications are acceptable
   for the intended project governance.
3. The test does not require paid Resource Groups.
4. The project owner explicitly accepts that state and any private audio stored
   in a free organization share its current `100GB` private allowance.
5. Batch stress testing confirms that fast validation does not lose state.

If Test A requires organization-wide `write`, the decision is not purely
technical: this model can support trusted research teams, but not isolated
arbitrary validators in a multi-project free organization.

## Most Likely Product Options After The Spike

| Option | Cost requirement | Isolation | Current assessment |
|---|---:|---|---|
| Free organization Bucket for a trusted team/project | None within current free quota | Organization-level only | Worth permission spike |
| One free organization for many unrelated projects | None within quota | Insufficient without paid Resource Groups | Not acceptable default |
| Team/Enterprise Resource Groups | Paid | Project/resource scoped | Secure optional tier only |
| Admin-provided external database | Depends on admin provider/free tier | Strongly project-owned when configured correctly | Remains fallback/default candidate |

## Official References

- Hugging Face organizations: <https://huggingface.co/docs/hub/organizations>
- Organization access control: <https://huggingface.co/docs/hub/organizations-security>
- Resource Groups: <https://huggingface.co/docs/hub/security-resource-groups>
- OAuth scopes and organization authorization: <https://huggingface.co/docs/hub/oauth>
- Storage Buckets: <https://huggingface.co/docs/hub/storage-buckets>
- Storage limits: <https://huggingface.co/docs/hub/storage-limits>
- Hub rate limits: <https://huggingface.co/docs/hub/rate-limits>
