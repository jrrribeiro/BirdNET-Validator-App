import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from huggingface_hub import CommitOperationAdd, HfApi

from src.domain.models import Project, Role


HF_PROJECT_STATE_BACKEND = "hf_project_store"
HF_PROJECT_STATE_SCHEMA_VERSION = 1


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

    def update_repo_visibility(
        self,
        repo_id: str,
        private: bool = False,
        *,
        token: str | None = None,
        repo_type: str | None = None,
    ) -> object: ...


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
            self._api.update_repo_visibility(
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
            if existing_files:
                sample = ", ".join(sorted(existing_files)[:5])
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
