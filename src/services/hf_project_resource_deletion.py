"""Permanent deletion helpers for administrator-owned Hugging Face project resources."""

from dataclasses import dataclass
from typing import Protocol

from huggingface_hub import HfApi

from src.domain.models import Project


class HfProjectResourceDeletionError(RuntimeError):
    """Raised when project resources cannot be safely deleted."""


class _DeletionApi(Protocol):
    def whoami(self, *, token: str | bool | None = None) -> dict[str, object]: ...

    def delete_repo(
        self,
        repo_id: str,
        *,
        token: str | bool | None = None,
        repo_type: str | None = None,
        missing_ok: bool = False,
    ) -> None: ...

    def delete_bucket(
        self,
        bucket_id: str,
        *,
        token: str | bool | None = None,
        missing_ok: bool = False,
    ) -> None: ...


@dataclass(frozen=True)
class HfProjectResourceDeletionResult:
    deleted_bucket_id: str | None
    deleted_state_repo_id: str | None
    deleted_dataset_repo_id: str


def _repo_namespace(repo_id: str) -> str:
    value = (repo_id or "").strip()
    if "/" not in value:
        raise HfProjectResourceDeletionError(f"Repository id '{value}' must be in owner/name format.")
    namespace, name = value.split("/", 1)
    if not namespace.strip() or not name.strip():
        raise HfProjectResourceDeletionError(f"Repository id '{value}' must be in owner/name format.")
    return namespace.strip()


def _ensure_same_namespace(*, label: str, resource_id: str, namespace: str) -> None:
    resource_namespace = _repo_namespace(resource_id)
    if resource_namespace.casefold() != namespace.casefold():
        raise HfProjectResourceDeletionError(
            f"Refusing to delete {label} '{resource_id}' because it is not in namespace '{namespace}'."
        )


class HfProjectResourceDeleter:
    """Delete the dataset, companion state repo, and validation Bucket for a project."""

    def __init__(self, api: _DeletionApi | None = None) -> None:
        self._api = api or HfApi()

    def delete_project_resources(
        self,
        *,
        project: Project,
        actor_username: str,
        storage_token: str | None,
        confirmed_slug: str,
        confirmation_checked: bool,
    ) -> HfProjectResourceDeletionResult:
        actor = (actor_username or "").strip()
        token = (storage_token or "").strip()
        expected_slug = (project.project_slug or "").strip()
        typed_slug = (confirmed_slug or "").strip()
        owner = (project.owner_username or "").strip()
        dataset_repo_id = (project.dataset_repo_id or "").strip()
        state_repo_id = (project.state_repo_id or "").strip() or None
        bucket_id = (project.validation_bucket_id or "").strip() or None

        if not actor:
            raise HfProjectResourceDeletionError("Login is required before deleting a project.")
        if not token:
            raise HfProjectResourceDeletionError(
                "Permanent deletion requires the configured administrator storage token."
            )
        if not expected_slug or typed_slug != expected_slug or not confirmation_checked:
            raise HfProjectResourceDeletionError(
                "Deletion was not confirmed. Type the exact project slug and check the confirmation box."
            )
        if not owner:
            raise HfProjectResourceDeletionError("Project owner is missing; deletion is blocked.")
        if actor.casefold() != owner.casefold():
            raise HfProjectResourceDeletionError("Only the project creator can permanently delete this project.")
        if not dataset_repo_id:
            raise HfProjectResourceDeletionError("Project dataset repository is missing; deletion is blocked.")

        namespace = _repo_namespace(dataset_repo_id)
        if namespace.casefold() != owner.casefold():
            raise HfProjectResourceDeletionError(
                "Refusing to delete project resources because the dataset namespace does not match the project owner."
            )
        if state_repo_id:
            _ensure_same_namespace(label="state repository", resource_id=state_repo_id, namespace=namespace)
        if bucket_id:
            _ensure_same_namespace(label="validation Bucket", resource_id=bucket_id, namespace=namespace)

        identity = self._api.whoami(token=token)
        token_owner = str(identity.get("name") or identity.get("fullname") or "").strip()
        if token_owner and token_owner.casefold() != namespace.casefold():
            raise HfProjectResourceDeletionError(
                "The configured storage token does not belong to the project owner namespace."
            )

        if bucket_id:
            self._api.delete_bucket(bucket_id, token=token, missing_ok=True)
        if state_repo_id:
            self._api.delete_repo(state_repo_id, repo_type="dataset", token=token, missing_ok=True)
        self._api.delete_repo(dataset_repo_id, repo_type="dataset", token=token, missing_ok=True)

        return HfProjectResourceDeletionResult(
            deleted_bucket_id=bucket_id,
            deleted_state_repo_id=state_repo_id,
            deleted_dataset_repo_id=dataset_repo_id,
        )
