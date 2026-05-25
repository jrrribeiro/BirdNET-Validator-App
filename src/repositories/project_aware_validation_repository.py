from collections.abc import Callable
from typing import Protocol

from src.domain.models import Project, Validation
from src.repositories.contracts import CurrentValidationRepository, ValidationEventRepository, ValidationRepository
from src.repositories.hf_bucket_validation_repository import (
    HF_BUCKET_VALIDATION_BACKEND,
    HfBucketValidationError,
    HfBucketValidationRepository,
)
from src.repositories.hf_project_state_validation_repository import HfProjectStateValidationError, HfProjectStateValidationRepository
from src.services.hf_project_state_store import HF_PROJECT_STATE_BACKEND


class _ValidationStateRepository(ValidationRepository, CurrentValidationRepository, ValidationEventRepository, Protocol):
    pass


class ProjectAwareValidationRepository:
    """Route validation state to a project-owned backend when explicitly enabled."""

    def __init__(
        self,
        *,
        fallback_repository: _ValidationStateRepository,
        project_lookup: Callable[[str], Project | None],
        token_provider: Callable[[Project], str | None],
        actor_token_provider: Callable[[Project, str], str | None] | None = None,
        enable_hf_project_state: bool = False,
        enable_hf_bucket_validations: bool = False,
        hf_repository_factory: Callable[[str, str], _ValidationStateRepository] | None = None,
        bucket_repository_factory: Callable[[str, str], _ValidationStateRepository] | None = None,
    ) -> None:
        self._fallback = fallback_repository
        self._project_lookup = project_lookup
        self._token_provider = token_provider
        self._actor_token_provider = actor_token_provider
        self._enable_hf_project_state = enable_hf_project_state
        self._enable_hf_bucket_validations = enable_hf_bucket_validations
        self._hf_repository_factory = hf_repository_factory or (
            lambda state_repo_id, token: HfProjectStateValidationRepository(
                state_repo_id=state_repo_id,
                token=token,
            )
        )
        self._bucket_repository_factory = bucket_repository_factory or (
            lambda bucket_id, token: HfBucketValidationRepository(
                bucket_id=bucket_id,
                token=token,
            )
        )
        self._hf_cache: dict[tuple[str, str], _ValidationStateRepository] = {}

    def _repository_for_project(
        self,
        project_slug: str,
        actor_username: str = "",
    ) -> _ValidationStateRepository:
        project = self._project_lookup(project_slug)
        if project is None:
            return self._fallback

        actor_token = ""
        if actor_username and self._actor_token_provider is not None:
            actor_token = (self._actor_token_provider(project, actor_username) or "").strip()

        if (project.validation_backend or "").strip() == HF_BUCKET_VALIDATION_BACKEND:
            if not self._enable_hf_bucket_validations:
                raise HfBucketValidationError(
                    "This project's validation state is stored in an HF Bucket, but Bucket validations are disabled "
                    "in this deployment. Enable the Bucket backend before reading or writing validations."
                )
            if not (project.validation_bucket_id or "").strip():
                raise HfBucketValidationError(
                    "This project declares HF Bucket validation storage but has no validation bucket id."
                )
            token = actor_token if actor_username else (self._token_provider(project) or "").strip()
            if not token:
                raise HfBucketValidationError(
                    "Bucket-backed validation requires the signed-in validator's Hugging Face authorization."
                )
            bucket_id = (project.validation_bucket_id or "").strip()
            cache_key = (f"bucket:{bucket_id}", token)
            repository = self._hf_cache.get(cache_key)
            if repository is None:
                repository = self._bucket_repository_factory(bucket_id, token)
                self._hf_cache[cache_key] = repository
            return repository

        token = actor_token if actor_username else (self._token_provider(project) or "").strip()
        if not token:
            if (
                actor_username
                and self._enable_hf_project_state
                and (project.state_backend or "").strip() == HF_PROJECT_STATE_BACKEND
                and (project.state_repo_id or "").strip()
            ):
                raise HfProjectStateValidationError(
                    "Private project-state validation requires the signed-in validator's Hugging Face authorization."
                )
            return self._fallback

        if not self._enable_hf_project_state:
            return self._fallback
        if (project.state_backend or "").strip() != HF_PROJECT_STATE_BACKEND:
            return self._fallback

        state_repo_id = (project.state_repo_id or "").strip()
        if not state_repo_id:
            return self._fallback

        cache_key = (f"repo:{state_repo_id}", token)
        repository = self._hf_cache.get(cache_key)
        if repository is None:
            repository = self._hf_repository_factory(state_repo_id, token)
            self._hf_cache[cache_key] = repository
        return repository

    def save_validation(self, project_slug: str, item: Validation, expected_version: int | None = None) -> int:
        return self._repository_for_project(project_slug, actor_username=item.validator).save_validation(
            project_slug=project_slug,
            item=item,
            expected_version=expected_version,
        )

    def load_current_snapshot(self, project_slug: str, actor_username: str = "") -> dict[str, dict[str, object]]:
        return self._repository_for_project(project_slug, actor_username=actor_username).load_current_snapshot(project_slug=project_slug)

    def list_events(self, project_slug: str, actor_username: str = "") -> list[dict[str, object]]:
        return self._repository_for_project(project_slug, actor_username=actor_username).list_events(project_slug=project_slug)

    def list_recent_events(
        self,
        project_slug: str,
        *,
        limit: int = 10,
        actor_username: str = "",
    ) -> list[dict[str, object]]:
        repository = self._repository_for_project(project_slug, actor_username=actor_username)
        reader = getattr(repository, "list_recent_events", None)
        if callable(reader):
            return reader(project_slug=project_slug, limit=limit)
        events = repository.list_events(project_slug=project_slug)
        return sorted(
            events,
            key=lambda event: str(event.get("timestamp") or event.get("created_at") or ""),
            reverse=True,
        )[: max(1, int(limit))]
