# HF Organization-Per-Project App Redesign Plan

Status: proposed target architecture after official API review and Bucket permission spike  
Created: 2026-05-26  
Target branch: `feature/project-state-security-privacy-review`

## Decision Summary

The target model is:

1. One Hugging Face organization represents one scientific validation project.
2. The project's private audio dataset, validation Bucket, manifest, and
   members belong to that organization.
3. The BirdNET Validator Space remains a central, reusable application outside
   project organizations and owns no project data.
4. Administrators manage projects primarily through the app, while the app
   uses supported Hugging Face organization, repository, and Bucket APIs.
5. For free organizations, organization creation and the initial invitation or
   role assignment of members remain Hugging Face web-interface steps because
   no supported free API flow was found for the app to perform them.

This model accepts that a trusted validator requires organization-level
`write` access within one project organization. It prevents that permission
from reaching unrelated scientific projects by isolating each project in a
separate organization.

## Evidence Already Collected

### Real Permission Spike In `ppbio-rabeca`

On 2026-05-25, the app investigation created:

```text
ppbio-rabeca/birdnet-validator-permission-spike
```

The following results were measured with separate user tokens:

| User | Organization role | Read private Bucket | Write private Bucket |
|---|---|---:|---:|
| `jrrribeiro` | `admin` | Passed | Passed |
| `jonathan2008r` | `write` | Passed | Passed |
| `jonathan2008r` | `contributor` | Passed | Failed with `403` |

Result: on a free organization, validators who persist decisions in an
administrator-created private Bucket must be trusted `write` members.

### Official HfApi / Hub API Review

The official Python API and the installed `huggingface_hub` client expose:

| Capability | Official support observed | Use in app |
|---|---:|---|
| Create private dataset repository in an organization namespace (`create_repo`) | Yes | Project initialization |
| Create private organization Bucket (`create_bucket`) | Yes | Validation state initialization |
| Read/write/list Bucket files (`batch_bucket_files`, `download_bucket_files`, `list_bucket_tree`) | Yes | Validation and recovery |
| List organization Buckets (`list_buckets(namespace=...)`) | Yes | Project discovery/health |
| Get organization overview (`get_organization_overview`) | Yes | Verify organization |
| List organization members (`list_organization_members`) | Yes | Render membership/readiness |
| Create organization | Not exposed in reviewed HfApi/docs | External HF onboarding step |
| Invite a non-member into an organization | Not provided in reviewed supported API guide | External HF onboarding step |
| Change an existing member's organization role by API | Official Hub endpoint, subscription required | Not available as free default |

The official **Programmatic User Access Control Management** guide documents:

```text
PUT /api/organizations/{org_name}/members/{username}/role
```

but requires:

1. an organization subscription plan, such as Team or Enterprise;
2. an authenticated organization member with `write` or `admin` permission;
3. a target user who is already a member of the organization; and
4. a fine-grained token with organization settings/member management access.

The endpoint returns `402` when the organization lacks the required
subscription. It therefore cannot be the default mechanism for free community
projects. The regular organization management documentation directs admins to
the Hugging Face Members settings interface to add members, send invites,
revoke membership, and change roles.

## Target Ownership Model

### Central App

```text
jrrribeiro/BirdNET-Validator-App
```

Responsibilities:

1. OAuth authentication and session-bound operations.
2. Project onboarding workflow.
3. Data loading, validation UI, dashboards, and exports.
4. Bucket state write batching, reconciliation, and auditing.

It must not durably own:

1. project audio;
2. detection tables;
3. project validation state;
4. project ACL as an alternate source of authorization; or
5. administrator or validator tokens.

### Project Organization

Example:

```text
ppbio-rabeca/
  PPBIO-RABECA                   private dataset repository
  PPBIO-RABECA-validation-state  private Storage Bucket
```

Recommended project resources:

| Resource | Location | Purpose | Mutable? |
|---|---|---|---:|
| Audio segments | Private dataset repository | Scientific source recordings | No during validation |
| Detection index/CSV | Same private dataset | Source observations and metadata | No during validation |
| `project.json` manifest | Validation Bucket metadata prefix initially | Project discovery/configuration | Infrequent admin update |
| Validation events | Validation Bucket `events/` | Append-oriented authored decisions | Yes |
| Reconciled snapshots | Validation Bucket `snapshots/` | Fast queue/progress reads | Yes |
| Audit archives/checkpoints | Validation Bucket `archives/` | Durable reconstruction/export | Append-oriented |

The original dataset remains versioned in a dataset repository. The frequently
mutating validation state uses a Bucket to avoid Git commit pressure.

## Organization Roles As Authorization

For the first organization-owned architecture:

| HF organization role | App role | Allowed UI capabilities |
|---|---|---|
| `admin` | Project administrator | Resource setup, integrity checks, member overview, validation, export, archival operations |
| `write` | Trusted validator | Read source dataset, write decisions, view progress, export as permitted |
| `read` | Viewer | Read project/progress only; no validation write |
| `contributor` | Viewer or unsupported state | Cannot write admin-created Bucket; never present as active validator |
| No membership/access | No project visibility | No project data access |

The app must query actual Hub access rather than treating a local invitation as
authorization. The current internal ACL/invite system becomes legacy migration
support only, then can be removed after organization mode is stable.

## User Experience And Supported Automation

The app should present an ordinary project workflow even though Hugging Face
resources are used underneath.

### Create Project Workspace

User-facing flow:

1. Admin clicks **Create Project**.
2. Admin supplies project name and the slug of an existing dedicated HF
   organization.
3. The app authenticates through OAuth and verifies the user has adequate
   access to initialize resources.
4. The app creates the private dataset repository and private validation
   Bucket in that organization.
5. The app writes the manifest and initial empty validation snapshot.
6. The app performs readiness checks and shows the created resource identifiers.

Supported automation:

| Step | App handles it? | Notes |
|---|---:|---|
| Validate existing organization | Yes | Hub API/client |
| Validate admin can create organization resources | Yes | Attempt/preflight and explicit errors |
| Create dataset | Yes | `create_repo(..., repo_type="dataset", private=True)` |
| Create Bucket | Yes | `create_bucket(..., private=True)` |
| Write initial manifest/state | Yes | Bucket API |
| Create the organization itself | No in free supported path | Link user to HF create organization page |

### Add Validators

The desired interface remains an app card named **Project Members**, but it
mirrors Hub authority.

Supported free-tier flow:

1. App lists current members and translates HF roles into app capabilities.
2. For a new validator, app explains that `write` is required and provides a
   direct link/instructions to the project's organization Members page.
3. Admin invites/adds the user through Hugging Face.
4. Admin returns and clicks **Refresh members**; the app detects the member and
   runs a readiness test.

Not supported as a free default:

1. Sending the actual organization membership invite through a supported API;
2. changing a member to `write` programmatically; or
3. removing organization membership programmatically.

Optional future paid feature:

If a project organization has a qualifying subscription, the app could support
role changes for existing members through the documented Hub endpoint, with a
separately authorized fine-grained organization-management credential. This
must never be required for free use.

## OAuth Boundary Still To Prove

The token spike proves Hub organization permission behavior, not the Space
OAuth permission granted to this app.

Official OAuth documentation confirms:

1. OAuth scopes can apply to organization resources selected by a user at
   authorization time.
2. `read-repos`, `write-repos`, and `manage-repos` descriptions refer to
   repositories; no reviewed scope description explicitly promises Bucket
   write behavior.
3. An app may request a specific organization through `orgIds=ORG_ID` in its
   authorization URL.

Therefore organization mode must not be implemented as production workflow
until a Space diagnostic proves:

| Actor | Required proof using OAuth only |
|---|---|
| Admin | Detect organization, read/create project resources or access test resources, write diagnostic Bucket marker |
| Validator `write` | Read private dataset, read Bucket, write diagnostic Bucket marker |
| Validator `read` or `contributor` | Cannot submit validation state |

No manual token may be entered for that proof.

## Data Safety Requirements

Because validators with `write` are trusted but technically broad within one
project organization:

1. One organization must contain only one scientific project.
2. Source audio/detection dataset must remain Git-backed and treated as
   immutable once validation begins.
3. App validation flows must never issue dataset writes for validators.
4. Bucket decisions must include actor, timestamp, detection key, decision,
   prior version, new version, and event ID.
5. Individual validation events remain auditable until compacted into
   append-oriented archives.
6. Snapshots are derived acceleration artifacts, not the sole source of truth.
7. Admin receives export/checkpoint tools and an integrity health view.
8. The uploader must retain local/source backups before moving real datasets
   into organization ownership.

## Required App Refactor

### Phase 0 - OAuth Organization/Bucket Capability Proof

Goal: close the one unresolved technical permission question.

Deliverables:

1. Add an experimental diagnostics card available only in the feature branch.
2. Use the active OAuth session, never manually entered tokens.
3. Configure the existing test organization and Bucket:
   `ppbio-rabeca/birdnet-validator-permission-spike`.
4. Show read/write results and clear failure interpretation.
5. Test both `jrrribeiro` (`admin`) and `jonathan2008r` restored to `write`.

Exit gate: both accounts write distinct diagnostic markers through OAuth.

### Phase 1 - Organization Project Domain Model

Goal: model project ownership around an organization and resource references.

Deliverables:

1. New manifest schema containing organization, dataset repo id, validation
   Bucket id, visibility, schema version, and initialized timestamp.
2. `OrganizationProjectService` that loads the manifest and verifies resource
   availability.
3. Membership mapper converting HF member role to application capability.
4. Legacy internal-ACL projects continue reading until migrated; no silent
   conversion.

### Phase 2 - Project Creation Wizard

Goal: keep the current easy create-project interaction while making the
organization the durable project owner.

Deliverables:

1. Request existing organization slug and project display name.
2. Check organization/resource permissions using OAuth.
3. Create private dataset and private Bucket automatically.
4. Initialize manifest and snapshots automatically.
5. Provide explicit incomplete-initialization recovery if any sub-step fails.
6. Do not store project tokens.

### Phase 3 - Membership-Based Access UX

Goal: replace misleading application invites with real HF readiness.

Deliverables:

1. Replace **User access and invitations** with **Project Members**.
2. List organization members and mapped app roles.
3. Show access requirements: validators must be trusted HF `write` members.
4. Provide a guided link to the HF Members page for add/change/remove actions
   in free organizations.
5. Add **Refresh members** and per-member readiness diagnostics.
6. Hide validation/project resources from non-members.

### Phase 4 - Bucket-Backed Validation As Primary Store

Goal: use already-developed fast mutable storage behavior with project-owned
organization Bucket routing.

Deliverables:

1. Route validation event writes to the organization Bucket using acting user's
   OAuth token.
2. Route snapshots, conflict recovery, progress, and CSV/XLSX exports from
   Bucket state.
3. Enable bounded compaction/audit archives after load testing.
4. Never fall back to filesystem, Supabase, or admin credentials when an
   organization write fails.

### Phase 5 - Dataset/Uploader Integration

Goal: allow administrators to provision the project in the app and send audio
to the created organization dataset efficiently.

Deliverables:

1. Surface the created dataset repo id for the uploader.
2. Update uploader/app integration documentation for organization destinations.
3. Preserve uploader index structure used by validation loading.
4. Add connection flow for existing organization datasets.

### Phase 6 - Migration And Removal Of Legacy Authorization

Goal: retire fragile centrally owned state after qualification.

Deliverables:

1. Migration preview for current Supabase/filesystem/experimental `_state`
   projects.
2. State import into organization Bucket with counts and verification report.
3. Remove project token entry from normal UI.
4. Disable internal invites/ACL as authority in organization mode.
5. Keep legacy mode read-only or explicitly marked until users migrate.

## Testing Program

### Permission And Security Tests

1. OAuth admin Bucket write in Space.
2. OAuth `write` validator Bucket write in Space.
3. OAuth `read`/`contributor` user validation write rejected.
4. User outside organization cannot see dataset or Bucket data.
5. App stores no OAuth/admin tokens in manifest, Bucket events, logs, or
   exports.

### Reliability Tests

1. Two validators write concurrently.
2. Duplicate submit is idempotent.
3. Snapshot conflict is recovered from authored events.
4. Restart/redeploy does not lose validation state.
5. Export reconstructs detections plus partial or completed validations.

### Scale Tests

1. Simulate decisions at approximately one action every three seconds per
   validator.
2. Run multiple concurrent validators.
3. Test event archive/compaction windows at realistic project sizes.
4. Measure Bucket latency and Hub rate-limit responses.
5. Confirm no dataset commits are generated by validation activity.

## Implementation Gate

The next code change should be **Phase 0 only**. Do not begin the onboarding
refactor until the existing organization Bucket succeeds through Space OAuth
for both administrator and `write` validator.

## Official References

- HfApi Client: <https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api>
- Storage Buckets: <https://huggingface.co/docs/hub/storage-buckets>
- Python Bucket guide: <https://huggingface.co/docs/huggingface_hub/guides/buckets>
- Organization management: <https://huggingface.co/docs/hub/organizations-managing>
- Organization access control: <https://huggingface.co/docs/hub/organizations-security>
- Programmatic User Access Control Management: <https://huggingface.co/docs/hub/programmatic-user-access-control>
- OAuth / Sign in with HF: <https://huggingface.co/docs/hub/oauth>
