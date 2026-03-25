from typing import Protocol

from src.domain.models import Detection, Project, User, Validation


class ProjectRepository(Protocol):
    def get_project(self, project_slug: str) -> Project: ...


class DetectionRepository(Protocol):
    def list_detections(self, project_slug: str, page: int, page_size: int) -> list[Detection]: ...


class ValidationRepository(Protocol):
    def save_validation(self, project_slug: str, item: Validation) -> None: ...


class AuthRepository(Protocol):
    def authenticate(self, username: str) -> User: ...
