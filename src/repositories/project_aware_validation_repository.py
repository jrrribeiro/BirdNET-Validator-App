from collections.abc import Callable
from typing import Protocol

from src.domain.models import Project, Validation
from src.repositories.contracts import CurrentValidationRepository, ValidationEventRepository, ValidationRepository
from src.repositories.hf_project_state_validation_repository import HfProjectStateValidationRepository
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
        enable_hf_project_state: bool = False,
        hf_repository_factory: Callable[[str, str], _ValidationStateRepository] | None = None,
    ) -> None:
        self._fallback = fallback_repository
        self._project_lookup = project_lookup
        self._token_provider = token_provider
        self._enable_hf_project_state = enable_hf_project_state
        self._hf_repository_factory = hf_repository_factory or (
            lambda state_repo_id, token: HfProjectStateValidationRepository(
                state_repo_id=state_repo_id,
                token=token,
            )
        )
        self._hf_cache: dict[tuple[str, str], _ValidationStateRepository] = {}

    def _repository_for_project(
        self,
        project_slug: str,
    ) -> _ValidationStateRepository:
        if not self._enable_hf_project_state:
            return self._fallback

        project = self._project_lookup(project_slug)
        if project is None:
            return self._fallback
        if (project.state_backend or "").strip() != HF_PROJECT_STATE_BACKEND:
            return self._fallback

        state_repo_id = (project.state_repo_id or "").strip()
        if not state_repo_id:
            return self._fallback

        token = (self._token_provider(project) or "").strip()
        if not token:
            return self._fallback

        cache_key = (state_repo_id, token)
        repository = self._hf_cache.get(cache_key)
        if repository is None:
            repository = self._hf_repository_factory(state_repo_id, token)
            self._hf_cache[cache_key] = repository
        return repository

    def save_validation(self, project_slug: str, item: Validation, expected_version: int | None = None) -> int:
        return self._repository_for_project(project_slug).save_validation(
            project_slug=project_slug,
            item=item,
            expected_version=expected_version,
        )

    def load_current_snapshot(self, project_slug: str) -> dict[str, dict[str, object]]:
        return self._repository_for_project(project_slug).load_current_snapshot(project_slug=project_slug)

    def list_events(self, project_slug: str) -> list[dict[str, object]]:
        return self._repository_for_project(project_slug).list_events(project_slug=project_slug)
