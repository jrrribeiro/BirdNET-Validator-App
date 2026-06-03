from typing import Protocol

from src.domain.models import Detection, Project, Role, Validation


class DetectionRepository(Protocol):
    def list_detections(
        self,
        project_slug: str,
        page: int,
        page_size: int,
        scientific_name: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
    ) -> list[Detection]: ...

    def count_detections(
        self,
        project_slug: str,
        scientific_name: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
    ) -> int: ...


class ValidationRepository(Protocol):
    def save_validation(self, project_slug: str, item: Validation, expected_version: int | None = None) -> int: ...


class ProjectCatalogRepository(Protocol):
    def load_projects(self) -> list[Project]: ...


class ProjectAccessRepository(Protocol):
    def load_user_access(self) -> dict[str, dict[str, Role]]: ...


class ProjectInviteRepository(Protocol):
    def load_pending_invites(self) -> dict[str, dict[str, dict[str, str]]]: ...


class BootstrapStateRepository(ProjectCatalogRepository, ProjectAccessRepository, ProjectInviteRepository, Protocol):
    def persist(
        self,
        projects: list[dict[str, object]],
        user_access: dict[str, dict[str, str]],
        invites: dict[str, dict[str, dict[str, str]]],
        *,
        allowed_removed_project_slugs: set[str] | None = None,
    ) -> None: ...


class ValidationEventRepository(Protocol):
    def list_events(self, project_slug: str, actor_username: str = "") -> list[dict[str, object]]: ...


class CurrentValidationRepository(Protocol):
    def load_current_snapshot(self, project_slug: str, actor_username: str = "") -> dict[str, dict[str, object]]: ...


class ProjectStateBackend(
    BootstrapStateRepository,
    ValidationRepository,
    ValidationEventRepository,
    CurrentValidationRepository,
    Protocol,
):
    """Combined persistence contract for a project-owned state backend."""


