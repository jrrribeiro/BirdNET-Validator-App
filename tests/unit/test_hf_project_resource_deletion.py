import pytest

from src.domain.models import Project
from src.services.hf_project_resource_deletion import (
    HfProjectResourceDeleter,
    HfProjectResourceDeletionError,
)


class FakeDeletionApi:
    def __init__(self, username: str = "owner") -> None:
        self.username = username
        self.calls: list[tuple[str, str]] = []

    def whoami(self, *, token=None):  # noqa: ANN001
        assert token == "hf_storage"
        return {"name": self.username}

    def delete_bucket(self, bucket_id: str, *, token=None, missing_ok: bool = False):  # noqa: ANN001
        assert token == "hf_storage"
        assert missing_ok is True
        self.calls.append(("bucket", bucket_id))

    def delete_repo(self, repo_id: str, *, token=None, repo_type=None, missing_ok: bool = False):  # noqa: ANN001
        assert token == "hf_storage"
        assert repo_type == "dataset"
        assert missing_ok is True
        self.calls.append(("repo", repo_id))


def _project() -> Project:
    return Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="owner/project-a",
        owner_username="owner",
        state_repo_id="owner/project-a_state",
        validation_bucket_id="owner/project-a_validation_state",
    )


def test_delete_project_resources_deletes_bucket_state_and_dataset_in_safe_order() -> None:
    api = FakeDeletionApi()
    result = HfProjectResourceDeleter(api=api).delete_project_resources(
        project=_project(),
        actor_username="owner",
        storage_token="hf_storage",
        confirmed_slug="project-a",
        confirmation_checked=True,
    )

    assert api.calls == [
        ("bucket", "owner/project-a_validation_state"),
        ("repo", "owner/project-a_state"),
        ("repo", "owner/project-a"),
    ]
    assert result.deleted_bucket_id == "owner/project-a_validation_state"
    assert result.deleted_state_repo_id == "owner/project-a_state"
    assert result.deleted_dataset_repo_id == "owner/project-a"


def test_delete_project_resources_requires_exact_confirmation() -> None:
    api = FakeDeletionApi()
    with pytest.raises(HfProjectResourceDeletionError, match="not confirmed"):
        HfProjectResourceDeleter(api=api).delete_project_resources(
            project=_project(),
            actor_username="owner",
            storage_token="hf_storage",
            confirmed_slug="wrong",
            confirmation_checked=True,
        )

    assert api.calls == []


def test_delete_project_resources_blocks_non_owner_actor() -> None:
    api = FakeDeletionApi()
    with pytest.raises(HfProjectResourceDeletionError, match="Only the project creator"):
        HfProjectResourceDeleter(api=api).delete_project_resources(
            project=_project(),
            actor_username="validator",
            storage_token="hf_storage",
            confirmed_slug="project-a",
            confirmation_checked=True,
        )

    assert api.calls == []


def test_delete_project_resources_blocks_cross_namespace_resources() -> None:
    api = FakeDeletionApi()
    project = _project().model_copy(update={"state_repo_id": "other/project-a_state"})

    with pytest.raises(HfProjectResourceDeletionError, match="namespace"):
        HfProjectResourceDeleter(api=api).delete_project_resources(
            project=project,
            actor_username="owner",
            storage_token="hf_storage",
            confirmed_slug="project-a",
            confirmation_checked=True,
        )

    assert api.calls == []
