import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from src.domain.models import Project, Role


HF_PROJECT_STATE_BACKEND = "hf_project_store"
HF_PROJECT_STATE_SCHEMA_VERSION = 1
HF_PROJECT_STATE_REPO_SCAFFOLDING_FILES = {".gitattributes"}


class HfProjectStateStoreError(Exception):
    """Raised when a Hugging Face project state store cannot be initialized."""


class HfProjectStateApi(Protocol):
    def create_repo(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        private: bool | None = None,
        repo_type: str | None = None,
        exist_ok: bool = False,
    ) -> object: ...

    def create_commit(
        self,
        repo_id: str,
        operations: list[CommitOperationAdd],
        *,
        commit_message: str,
        token: str | None = None,
        repo_type: str | None = None,
    ) -> object: ...

    def list_repo_files(
        self,
        repo_id: str,
        *,
        repo_type: str | None = None,
        token: str | None = None,
    ) -> list[str]: ...

    # NOTE: huggingface_hub renamed visibility APIs over time.
    # We call `update_repo_settings(private=...)` when available, and fall back to older names.
    def update_repo_settings(
        self,
        repo_id: str,
        *,
        private: bool | None = None,
        token: str | None = None,
        repo_type: str | None = None,
    ) -> object: ...

    def update_repo_visibility(
        self,
        repo_id: str,
        private: bool = False,
        *,
        token: str | None = None,
        repo_type: str | None = None,
    ) -> object: ...

    def set_repo_visibility(
        self,
        repo_id: str,
        private: bool = False,
        *,
        token: str | None = None,
        repo_type: str | None = None,
    ) -> object: ...


class HfProjectStateReadApi(Protocol):
    def read_text(
        self,
        repo_id: str,
        path_in_repo: str,
        *,
        repo_type: str | None = None,
        token: str | None = None,
    ) -> str: ...


@dataclass(frozen=True)
class HfProjectStateStoreInitResult:
    state_repo_id: str
    manifest: dict[str, object]
    initialized: bool
    reused_existing: bool


@dataclass(frozen=True)
class HfProjectStateStoreSyncResult:
    state_repo_id: str
    synced_paths: list[str]


@dataclass(frozen=True)
class HfProjectStatePermissionProbeResult:
    state_repo_id: str
    actor_username: str
    diagnostic_path: str
    verified_at: str


@dataclass(frozen=True)
class HfProjectStateStoreLoadedProject:
    state_repo_id: str
    project: Project
    user_access: dict[str, dict[str, Role]]
    pending_invites: dict[str, dict[str, dict[str, str]]]


def default_state_repo_id(dataset_repo_id: str) -> str:
    repo_id = (dataset_repo_id or "").strip().strip("/")
    if "/" not in repo_id:
        raise HfProjectStateStoreError("Dataset repo id must be in owner/name format to create a companion state repo.")
    namespace, name = repo_id.split("/", 1)
    namespace = namespace.strip()
    name = name.strip()
    if not namespace or not name:
        raise HfProjectStateStoreError("Dataset repo id must be in owner/name format to create a companion state repo.")
    if name.endswith("_state"):
        state_name = name
    else:
        state_name = f"{name}_state"
    return f"{namespace}/{state_name}"


def _project_json(
    project: Project,
    state_repo_id: str,
    created_at: str,
    *,
    updated_at: str | None = None,
    archived_at: str | None = None,
) -> dict[str, object]:
    return {
        "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
        "project_id": project.project_id,
        "project_slug": project.project_slug,
        "project_name": project.name,
        "dataset_repo_id": project.dataset_repo_id,
        "state_backend": HF_PROJECT_STATE_BACKEND,
        "state_ref": state_repo_id,
        "state_schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
        "state_status": "archived" if archived_at else "ready",
        "validation_backend": project.validation_backend,
        "validation_bucket_id": project.validation_bucket_id,
        "owner_username": project.owner_username,
        "visibility": project.visibility,
        "active": bool(project.active) and archived_at is None,
        "created_at": created_at,
        "updated_at": updated_at or created_at,
        **({"archived_at": archived_at} if archived_at else {}),
    }


def _acl_json(project: Project, creator_username: str, created_at: str) -> dict[str, object]:
    return {
        "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
        "project_slug": project.project_slug,
        "updated_at": created_at,
        "users": {
            creator_username: {
                "role": Role.admin.value,
                "active": True,
                "granted_at": created_at,
                "granted_by": creator_username,
            }
        },
    }


def _readme(project: Project, state_repo_id: str) -> str:
    title = re.sub(r"\s+", " ", project.name).strip() or project.project_slug
    return (
        f"# {title} validator state\n\n"
        "Private companion dataset for BirdNET Validator project state.\n\n"
        f"- Project: `{project.project_slug}`\n"
        f"- Audio dataset: `{project.dataset_repo_id}`\n"
        f"- State repo: `{state_repo_id}`\n\n"
        "This repository stores validation state, ACL metadata, invites, snapshots, "
        "and future recovery artifacts for the validator app. It should remain private "
        "unless the project admin intentionally changes the visibility policy.\n"
    )


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")


def _acl_from_access_map(
    *,
    project: Project,
    user_access: dict[str, dict[str, str]],
    updated_at: str,
    actor_username: str,
) -> dict[str, object]:
    users: dict[str, dict[str, object]] = {}
    for username, roles_by_project in sorted((user_access or {}).items()):
        if not isinstance(roles_by_project, dict):
            continue
        role = str(roles_by_project.get(project.project_slug) or "").strip().lower()
        if role not in {Role.admin.value, Role.validator.value}:
            continue
        users[str(username)] = {
            "role": role,
            "active": True,
            "updated_at": updated_at,
            "updated_by": actor_username or None,
        }

    return {
        "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
        "project_slug": project.project_slug,
        "updated_at": updated_at,
        "users": users,
    }


def _invites_from_invite_map(
    *,
    project: Project,
    pending_invites: dict[str, dict[str, dict[str, str]]],
    updated_at: str,
) -> dict[str, object]:
    pending: dict[str, dict[str, object]] = {}
    for invite_key, invites_by_project in sorted((pending_invites or {}).items()):
        if not isinstance(invites_by_project, dict):
            continue
        invite = invites_by_project.get(project.project_slug)
        if not isinstance(invite, dict):
            continue
        pending[str(invite_key)] = {
            "role": str(invite.get("role") or "validator"),
            "invited_by": str(invite.get("invited_by") or ""),
            "created_at": str(invite.get("created_at") or ""),
            "expires_at": str(invite.get("expires_at") or ""),
            "username": str(invite.get("username") or ""),
            "invitee_email": str(invite.get("invitee_email") or ""),
        }

    return {
        "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
        "project_slug": project.project_slug,
        "updated_at": updated_at,
        "pending": pending,
    }


def _require_supported_schema(payload: dict[str, object], label: str) -> None:
    try:
        schema_version = int(payload.get("schema_version") or 1)
    except Exception:
        schema_version = 1
    if schema_version > HF_PROJECT_STATE_SCHEMA_VERSION:
        raise HfProjectStateStoreError(
            f"{label} uses schema_version={schema_version}, but this app supports up to {HF_PROJECT_STATE_SCHEMA_VERSION}."
        )


def _project_from_project_payload(payload: dict[str, object], state_repo_id: str) -> Project | None:
    _require_supported_schema(payload, "project.json")
    slug = str(payload.get("project_slug") or "").strip()
    if not slug:
        raise HfProjectStateStoreError("project.json is missing project_slug.")

    active = bool(payload.get("active", True))
    state_status = str(payload.get("state_status") or "ready").strip() or "ready"
    if state_status == "archived":
        active = False
    if not active:
        return None

    project_id = str(payload.get("project_id") or "").strip()
    if not project_id:
        project_id = str(uuid5(NAMESPACE_URL, f"birdnet-validator:{slug}"))
    if len(project_id) < 8:
        project_id = str(uuid4())

    state_schema_version = int(payload.get("state_schema_version") or HF_PROJECT_STATE_SCHEMA_VERSION)
    return Project(
        project_id=project_id,
        project_slug=slug,
        name=str(payload.get("project_name") or payload.get("name") or slug).strip() or slug,
        dataset_repo_id=str(payload.get("dataset_repo_id") or "").strip(),
        visibility=str(payload.get("visibility") or "collaborative").strip() or "collaborative",
        owner_username=(str(payload.get("owner_username") or "").strip() or None),
        dataset_token=None,
        state_backend=HF_PROJECT_STATE_BACKEND,
        state_repo_id=state_repo_id,
        state_schema_version=state_schema_version,
        state_status=state_status,
        validation_backend=str(payload.get("validation_backend") or "app_backend").strip() or "app_backend",
        validation_bucket_id=(str(payload.get("validation_bucket_id") or "").strip() or None),
        active=True,
    )


def _access_from_acl_payload(payload: dict[str, object], project_slug: str) -> dict[str, dict[str, Role]]:
    _require_supported_schema(payload, "acl.json")
    users = payload.get("users") if isinstance(payload, dict) else {}
    if not isinstance(users, dict):
        return {}

    result: dict[str, dict[str, Role]] = {}
    for username, raw_access in users.items():
        user = str(username or "").strip()
        if not user:
            continue
        if isinstance(raw_access, dict):
            if not bool(raw_access.get("active", True)):
                continue
            role_text = str(raw_access.get("role") or "").strip().lower()
        else:
            role_text = str(raw_access or "").strip().lower()
        if role_text not in {Role.admin.value, Role.validator.value}:
            continue
        result.setdefault(user, {})[project_slug] = Role(role_text)
    return result


def _invites_from_payload(payload: dict[str, object], project_slug: str) -> dict[str, dict[str, dict[str, str]]]:
    _require_supported_schema(payload, "invites.json")
    pending = payload.get("pending") if isinstance(payload, dict) else {}
    if not isinstance(pending, dict):
        return {}

    result: dict[str, dict[str, dict[str, str]]] = {}
    for invite_key, raw_invite in pending.items():
        key = str(invite_key or "").strip()
        if not key or not isinstance(raw_invite, dict):
            continue
        invite_payload = raw_invite.get(project_slug) if isinstance(raw_invite.get(project_slug), dict) else raw_invite
        if not isinstance(invite_payload, dict):
            continue
        role = str(invite_payload.get("role") or "validator").strip().lower()
        if role not in {Role.admin.value, Role.validator.value}:
            continue
        result.setdefault(key, {})[project_slug] = {
            "role": role,
            "invited_by": str(invite_payload.get("invited_by") or ""),
            "created_at": str(invite_payload.get("created_at") or ""),
            "expires_at": str(invite_payload.get("expires_at") or ""),
            "username": str(invite_payload.get("username") or ""),
            "invitee_email": str(invite_payload.get("invitee_email") or ""),
        }
    return result


class HfProjectStateStoreInitializer:
    def __init__(self, api: HfProjectStateApi | None = None) -> None:
        self._api = api or HfApi()

    def initialize(
        self,
        *,
        project: Project,
        creator_username: str,
        token: str | None,
        state_repo_id: str | None = None,
    ) -> HfProjectStateStoreInitResult:
        token_value = (token or "").strip()
        if not token_value:
            raise HfProjectStateStoreError("A Hugging Face token is required to create the private project state repository.")

        resolved_state_repo_id = (state_repo_id or "").strip() or default_state_repo_id(project.dataset_repo_id)
        created_at = datetime.now(UTC).isoformat()
        manifest = _project_json(project, resolved_state_repo_id, created_at)
        acl = _acl_json(project, creator_username, created_at)
        invites = {
            "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
            "project_slug": project.project_slug,
            "pending": {},
        }
        current = {
            "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
            "project_slug": project.project_slug,
            "items": {},
        }

        try:
            self._api.create_repo(
                repo_id=resolved_state_repo_id,
                token=token_value,
                private=True,
                repo_type="dataset",
                exist_ok=True,
            )
            # Enforce privacy even when the repo already exists.
            # huggingface_hub>=0.26: update_repo_settings(private=...)
            # older: update_repo_visibility / set_repo_visibility.
            if hasattr(self._api, "update_repo_settings"):
                self._api.update_repo_settings(
                    repo_id=resolved_state_repo_id,
                    private=True,
                    token=token_value,
                    repo_type="dataset",
                )
            elif hasattr(self._api, "update_repo_visibility"):
                self._api.update_repo_visibility(
                    repo_id=resolved_state_repo_id,
                    private=True,
                    token=token_value,
                    repo_type="dataset",
                )
            elif hasattr(self._api, "set_repo_visibility"):
                self._api.set_repo_visibility(
                    repo_id=resolved_state_repo_id,
                    private=True,
                    token=token_value,
                    repo_type="dataset",
                )
            existing_files = set(
                self._api.list_repo_files(
                    repo_id=resolved_state_repo_id,
                    repo_type="dataset",
                    token=token_value,
                )
            )
            if "project.json" in existing_files:
                return HfProjectStateStoreInitResult(
                    state_repo_id=resolved_state_repo_id,
                    manifest=manifest,
                    initialized=False,
                    reused_existing=True,
                )
            unexpected_files = existing_files - HF_PROJECT_STATE_REPO_SCAFFOLDING_FILES
            if unexpected_files:
                sample = ", ".join(sorted(unexpected_files)[:5])
                raise HfProjectStateStoreError(
                    "The companion state repo already contains files but no project.json manifest. "
                    f"Refusing to initialize automatically to avoid overwriting existing state. Existing files: {sample}"
                )
            self._api.create_commit(
                repo_id=resolved_state_repo_id,
                repo_type="dataset",
                token=token_value,
                commit_message="Initialize BirdNET Validator state store",
                operations=[
                    CommitOperationAdd(
                        path_in_repo="README.md",
                        path_or_fileobj=_readme(project, resolved_state_repo_id).encode("utf-8"),
                    ),
                    CommitOperationAdd(
                        path_in_repo="project.json",
                        path_or_fileobj=json.dumps(manifest, indent=2).encode("utf-8"),
                    ),
                    CommitOperationAdd(
                        path_in_repo="acl.json",
                        path_or_fileobj=json.dumps(acl, indent=2).encode("utf-8"),
                    ),
                    CommitOperationAdd(
                        path_in_repo="invites.json",
                        path_or_fileobj=json.dumps(invites, indent=2).encode("utf-8"),
                    ),
                    CommitOperationAdd(
                        path_in_repo="snapshots/current.json",
                        path_or_fileobj=json.dumps(current, indent=2).encode("utf-8"),
                    ),
                    CommitOperationAdd(
                        path_in_repo="events/.gitkeep",
                        path_or_fileobj=b"",
                    ),
                ],
            )
        except Exception as exc:
            raise HfProjectStateStoreError(f"Could not initialize private HF state repo {resolved_state_repo_id}: {exc}") from exc

        return HfProjectStateStoreInitResult(
            state_repo_id=resolved_state_repo_id,
            manifest=manifest,
            initialized=True,
            reused_existing=False,
        )


class HfProjectStateStoreLoader:
    def __init__(self, api: HfProjectStateReadApi | None = None) -> None:
        self._api = api

    def _read_text(self, repo_id: str, path_in_repo: str, token: str | None) -> str:
        token_value = (token or "").strip() or None
        if self._api is not None:
            return self._api.read_text(
                repo_id=repo_id,
                path_in_repo=path_in_repo,
                repo_type="dataset",
                token=token_value,
            )

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=token_value,
        )
        return Path(local_path).read_text(encoding="utf-8")

    def _read_json(self, repo_id: str, path_in_repo: str, token: str | None, default: object | None = None) -> object:
        try:
            raw_text = self._read_text(repo_id, path_in_repo, token)
        except Exception as exc:
            if default is not None:
                return default
            raise HfProjectStateStoreError(f"Could not read {path_in_repo} from {repo_id}: {exc}") from exc

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise HfProjectStateStoreError(f"Could not parse {path_in_repo} from {repo_id}: {exc}") from exc

    def load_project_state(
        self,
        *,
        state_repo_id: str,
        token: str | None,
    ) -> HfProjectStateStoreLoadedProject | None:
        repo_id = (state_repo_id or "").strip()
        if not repo_id:
            raise HfProjectStateStoreError("A project state repo id is required.")

        project_payload = self._read_json(repo_id, "project.json", token)
        if not isinstance(project_payload, dict):
            raise HfProjectStateStoreError(f"project.json in {repo_id} must be a JSON object.")

        project = _project_from_project_payload(project_payload, repo_id)
        if project is None:
            return None

        acl_payload = self._read_json(
            repo_id,
            "acl.json",
            token,
            default={
                "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
                "project_slug": project.project_slug,
                "users": {},
            },
        )
        invites_payload = self._read_json(
            repo_id,
            "invites.json",
            token,
            default={
                "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
                "project_slug": project.project_slug,
                "pending": {},
            },
        )
        if not isinstance(acl_payload, dict):
            acl_payload = {"schema_version": HF_PROJECT_STATE_SCHEMA_VERSION, "users": {}}
        if not isinstance(invites_payload, dict):
            invites_payload = {"schema_version": HF_PROJECT_STATE_SCHEMA_VERSION, "pending": {}}

        return HfProjectStateStoreLoadedProject(
            state_repo_id=repo_id,
            project=project,
            user_access=_access_from_acl_payload(acl_payload, project.project_slug),
            pending_invites=_invites_from_payload(invites_payload, project.project_slug),
        )


class HfProjectStateStoreConnector:
    """Loads an existing project state only for an authorized project admin."""

    def __init__(self, loader: HfProjectStateStoreLoader | None = None) -> None:
        self._loader = loader or HfProjectStateStoreLoader()

    def connect_admin_project(
        self,
        *,
        state_repo_id: str,
        token: str | None,
        actor_username: str,
    ) -> HfProjectStateStoreLoadedProject:
        token_value = (token or "").strip()
        actor = (actor_username or "").strip()
        if not token_value:
            raise HfProjectStateStoreError(
                "Sign in with your Hugging Face account or token before connecting a private project state repository."
            )
        if not actor:
            raise HfProjectStateStoreError("An authenticated Hugging Face identity is required to connect a project.")

        loaded = self._loader.load_project_state(state_repo_id=state_repo_id, token=token_value)
        if loaded is None:
            raise HfProjectStateStoreError("This project state is archived and cannot be connected as an active project.")

        role = loaded.user_access.get(actor, {}).get(loaded.project.project_slug)
        if role != Role.admin:
            raise HfProjectStateStoreError(
                "Only an ADMIN recorded in this project's ACL can connect its private state repository."
            )

        owner = (loaded.project.owner_username or "").strip()
        if loaded.project.visibility == "private" and owner and owner != actor:
            raise HfProjectStateStoreError("Only the owner can connect a private project state repository.")
        return loaded


class HfProjectStatePermissionProbe:
    """Prove that the acting OAuth identity can read and write a private state repo."""

    def __init__(
        self,
        *,
        loader: HfProjectStateStoreLoader | None = None,
        api: HfProjectStateApi | None = None,
    ) -> None:
        self._loader = loader or HfProjectStateStoreLoader()
        self._api = api or HfApi()

    def probe(
        self,
        *,
        project: Project,
        actor_username: str,
        token: str | None,
    ) -> HfProjectStatePermissionProbeResult:
        actor = (actor_username or "").strip()
        token_value = (token or "").strip()
        state_repo_id = (project.state_repo_id or "").strip()
        if not actor or not token_value:
            raise HfProjectStateStoreError(
                "Sign in with your Hugging Face OAuth account before testing private state authorization."
            )
        if not state_repo_id:
            raise HfProjectStateStoreError("This project has no private `_state` repository to test.")

        try:
            loaded = self._loader.load_project_state(state_repo_id=state_repo_id, token=token_value)
        except Exception as exc:
            raise HfProjectStateStoreError(
                f"State authorization read failed for {state_repo_id} using the signed-in account: {exc}"
            ) from exc
        if loaded is None or loaded.project.project_slug != project.project_slug:
            raise HfProjectStateStoreError("The private state manifest does not match the selected project.")

        verified_at = datetime.now(UTC).isoformat()
        diagnostic_id = str(uuid4())
        diagnostic_path = f"diagnostics/oauth-permission-proof/{diagnostic_id}.json"
        payload = {
            "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
            "diagnostic_type": "oauth_permission_proof",
            "project_slug": project.project_slug,
            "state_repo_id": state_repo_id,
            "actor_username": actor,
            "verified_at": verified_at,
            "diagnostic_id": diagnostic_id,
        }
        try:
            self._api.create_commit(
                repo_id=state_repo_id,
                repo_type="dataset",
                token=token_value,
                commit_message=f"Verify OAuth state access for {actor}",
                operations=[
                    CommitOperationAdd(path_in_repo=diagnostic_path, path_or_fileobj=_json_bytes(payload)),
                ],
            )
        except Exception as exc:
            raise HfProjectStateStoreError(
                f"State authorization write failed for {state_repo_id} using the signed-in account: {exc}"
            ) from exc

        return HfProjectStatePermissionProbeResult(
            state_repo_id=state_repo_id,
            actor_username=actor,
            diagnostic_path=diagnostic_path,
            verified_at=verified_at,
        )


class HfProjectStateStoreSync:
    def __init__(self, api: HfProjectStateApi | None = None) -> None:
        self._api = api or HfApi()

    def sync_project_state(
        self,
        *,
        project: Project,
        user_access: dict[str, dict[str, str]],
        pending_invites: dict[str, dict[str, dict[str, str]]],
        token: str | None,
        actor_username: str = "",
        archived: bool = False,
    ) -> HfProjectStateStoreSyncResult:
        token_value = (token or "").strip()
        if not token_value:
            raise HfProjectStateStoreError("A Hugging Face token is required to sync the private project state repository.")

        state_repo_id = (project.state_repo_id or "").strip() or default_state_repo_id(project.dataset_repo_id)
        updated_at = datetime.now(UTC).isoformat()
        archived_at = updated_at if archived else None
        project_payload = _project_json(
            project,
            state_repo_id,
            updated_at,
            updated_at=updated_at,
            archived_at=archived_at,
        )
        acl_payload = (
            {
                "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
                "project_slug": project.project_slug,
                "updated_at": updated_at,
                "users": {},
            }
            if archived
            else _acl_from_access_map(
                project=project,
                user_access=user_access,
                updated_at=updated_at,
                actor_username=actor_username,
            )
        )
        invites_payload = (
            {
                "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
                "project_slug": project.project_slug,
                "updated_at": updated_at,
                "pending": {},
            }
            if archived
            else _invites_from_invite_map(
                project=project,
                pending_invites=pending_invites,
                updated_at=updated_at,
            )
        )
        paths = ["project.json", "acl.json", "invites.json"]

        try:
            self._api.create_commit(
                repo_id=state_repo_id,
                repo_type="dataset",
                token=token_value,
                commit_message=(
                    "Archive BirdNET Validator project state"
                    if archived
                    else "Sync BirdNET Validator project state"
                ),
                operations=[
                    CommitOperationAdd(path_in_repo="project.json", path_or_fileobj=_json_bytes(project_payload)),
                    CommitOperationAdd(path_in_repo="acl.json", path_or_fileobj=_json_bytes(acl_payload)),
                    CommitOperationAdd(path_in_repo="invites.json", path_or_fileobj=_json_bytes(invites_payload)),
                ],
            )
        except Exception as exc:
            raise HfProjectStateStoreError(f"Could not sync private HF state repo {state_repo_id}: {exc}") from exc

        return HfProjectStateStoreSyncResult(state_repo_id=state_repo_id, synced_paths=paths)
