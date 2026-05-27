# HF Administrator-Owned Private Storage Delivery Plan

Status: immediate delivery architecture implemented on `feature/project-state-security-privacy-review`
Created: 2026-05-26
Purpose: begin real validation with a small trusted team while keeping unpublished data inaccessible outside the app.

## Decision

For the immediate workflow, all durable project resources belong to the project administrator's personal Hugging Face account:

```text
administrator namespace/
  audio-dataset                   private Dataset repository
  audio-dataset_state             private Dataset repository for project, ACL, and invites
  audio-dataset_validation_state  private Storage Bucket for validation events and snapshots
```

The deployed Space receives one protected administrator storage secret. Team members authenticate with their own Hugging Face identities, but the app uses their identity only for its persisted ACL and validation attribution. Authorized reads and writes to private project resources are performed by the backend with the storage secret.

This avoids organization membership and lets administrators invite or assign validators entirely inside the app.

## Security Boundary

This mode protects against external Hub users reading a private dataset, state repository, or Bucket. An invited validator does not receive direct Hub access and cannot browse those resources outside this app.

The trusted boundary is the deployed Space:

1. its owner controls the administrator storage secret;
2. its callbacks must validate app authorization before serving private audio or state;
3. trusted project validators can receive project audio through the app and submit decisions; and
4. compromise of the Space or its storage secret exposes projects accessible to that credential.

For public distribution to independent administrators, this model is not sufficient because the operator would hold their storage credentials. The planned distribution architecture remains administrator-owned Supabase with an Edge Function validating Hugging Face identity, documented in `TEMP_BYO_SUPABASE_EDGE_FUNCTION_HF_IDENTITY_PLAN.md`.

## Resource Roles

| Resource | Purpose | Writer |
|---|---|---|
| Private audio dataset | audio and source detection index | administrator/uploader |
| Private `_state` dataset | manifest, ACL, invitations, recovery metadata | app backend using administrator secret |
| Private validation Bucket | events, current snapshot, archives | app backend using administrator secret |
| Local Space disk | temporary cache and generated downloads | non-authoritative |

## Implemented Behavior

1. `BIRDNET_HF_ADMIN_STORAGE_MODE_ENABLED=true` activates this mode; configuring `BIRDNET_HF_STORAGE_TOKEN` also activates it automatically.
2. `BIRDNET_HF_STORAGE_TOKEN` is read server-side and never saved in project records or shown in the UI. `HF_TOKEN` remains a compatibility fallback.
3. New projects require a private dataset in the personal namespace belonging to the configured storage credential.
4. New projects automatically initialize a private validation Bucket and private companion `_state` repository.
5. Project manifests persist the Bucket reference; ACL and invite changes sync to `_state`.
6. Validation Bucket reads and writes route through the backend storage credential while retaining the actual validator username in each event.
7. Dataset metadata and audio fetches route through the backend credential only after server-side authorization of the session and project role.
8. At startup the app automatically discovers the administrator's private companion `*_state` datasets for rebuild recovery; a manual repo list remains an optional fallback.
9. This mode takes precedence over configured Supabase persistence so durable project state does not remain dependent on the app operator's Supabase database.
10. Empty `*_state` repositories left by interrupted setup attempts are ignored during discovery until they contain a `project.json` manifest.
11. The administrator can run a backend storage health check; validators do not test or receive direct Bucket permissions.

## Space Configuration

Configure:

```text
BIRDNET_HF_ADMIN_STORAGE_MODE_ENABLED=true
BIRDNET_AUTH_MODE=hf_token
```

Configure as a Secret:

```text
BIRDNET_HF_STORAGE_TOKEN=<administrator credential with required private dataset/repo/Bucket read-write access>
```

Use a scoped credential limited to the administrator's required resources when Hugging Face token controls permit it. Do not share it with validators or enter it in the app login input.

## Acceptance Gate Before Real Validation

1. Administrator creates a private test dataset in their personal namespace.
2. Administrator creates the app project; the app reports the generated private `_state` repository and validation Bucket.
3. Administrator restarts the Space without manually registering the new `_state` repository.
4. Project is discovered and restored with its ACL and Bucket reference intact.
5. Administrator assigns or invites a second Hugging Face account only through the app.
6. The second account logs in with its own identity, opens the project, loads private audio, and submits validations without Hub resource permissions.
7. The second account cannot access the private audio dataset, `_state`, or Bucket directly on Hugging Face.
8. Administrator exports CSV/XLSX and confirms the second user's validation attribution.
9. A rebuild restores the project and previous validations.
10. A short rapid-validation test confirms correctness under normal research throughput before full production use.

## Performance Note

Bucket events avoid one Git commit per validation. Before millions of validations or very high concurrency, test event compaction and snapshot behavior under sustained writes; if necessary, introduce additional snapshot sharding or buffered batches without changing this authorization model.
