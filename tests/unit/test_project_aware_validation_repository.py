from collections import defaultdict
from typing import DefaultDict

import pytest

from src.domain.models import Project, Validation
from src.repositories.hf_bucket_validation_repository import HF_BUCKET_VALIDATION_BACKEND, HfBucketValidationError
from src.repositories.hf_project_state_validation_repository import HfProjectStateValidationError
from src.repositories.project_aware_validation_repository import ProjectAwareValidationRepository
from src.services.hf_project_state_store import HF_PROJECT_STATE_BACKEND


class FakeValidationStateRepository:
    def __init__(self, label: str) -> None:
        self.label = label
        self.saved: list[tuple[str, str]] = []
        self.snapshots: DefaultDict[str, dict[str, dict[str, object]]] = defaultdict(dict)

    def save_validation(self, project_slug: str, item: Validation, expected_version: int | None = None) -> int:
        _ = expected_version
        self.saved.append((project_slug, item.detection_key))
        self.snapshots[project_slug][item.detection_key] = {
            "status": item.status,
            "version": len(self.saved),
        }
        return len(self.saved)

    def load_current_snapshot(self, project_slug: str) -> dict[str, dict[str, object]]:
        return dict(self.snapshots.get(project_slug, {}))

    def list_events(self, project_slug: str) -> list[dict[str, object]]:
        return [{"project_slug": slug, "detection_key": key} for slug, key in self.saved if slug == project_slug]


def _validation() -> Validation:
    return Validation(
        detection_key="audio-a-0000000001",
        status="positive",
        corrected_species=None,
        notes="",
        validator="validator-a",
    )


def test_project_aware_repository_uses_fallback_when_hf_state_disabled() -> None:
    fallback = FakeValidationStateRepository("fallback")
    hf_repo = FakeValidationStateRepository("hf")
    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="owner/project-a",
        state_backend=HF_PROJECT_STATE_BACKEND,
        state_repo_id="owner/project-a_state",
    )
    router = ProjectAwareValidationRepository(
        fallback_repository=fallback,
        project_lookup=lambda slug: project if slug == "project-a" else None,
        token_provider=lambda _: "hf_token",
        enable_hf_project_state=False,
        hf_repository_factory=lambda _repo, _token: hf_repo,
    )

    assert router.save_validation("project-a", _validation(), expected_version=0) == 1

    assert fallback.saved == [("project-a", "audio-a-0000000001")]
    assert hf_repo.saved == []


def test_project_aware_repository_routes_hf_project_state_when_enabled() -> None:
    fallback = FakeValidationStateRepository("fallback")
    hf_repo = FakeValidationStateRepository("hf")
    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="owner/project-a",
        state_backend=HF_PROJECT_STATE_BACKEND,
        state_repo_id="owner/project-a_state",
    )
    router = ProjectAwareValidationRepository(
        fallback_repository=fallback,
        project_lookup=lambda slug: project if slug == "project-a" else None,
        token_provider=lambda _: "hf_token",
        actor_token_provider=lambda _project, username: "hf_validator_token" if username == "validator-a" else None,
        enable_hf_project_state=True,
        hf_repository_factory=lambda _repo, _token: hf_repo,
    )

    assert router.save_validation("project-a", _validation(), expected_version=0) == 1

    assert fallback.saved == []
    assert hf_repo.saved == [("project-a", "audio-a-0000000001")]
    assert router.load_current_snapshot("project-a")["audio-a-0000000001"]["status"] == "positive"


def test_project_aware_repository_rejects_collaborator_write_without_personal_state_token() -> None:
    fallback = FakeValidationStateRepository("fallback")
    hf_repo = FakeValidationStateRepository("hf")
    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="owner/project-a",
        state_backend=HF_PROJECT_STATE_BACKEND,
        state_repo_id="owner/project-a_state",
    )
    router = ProjectAwareValidationRepository(
        fallback_repository=fallback,
        project_lookup=lambda slug: project if slug == "project-a" else None,
        token_provider=lambda _: None,
        enable_hf_project_state=True,
        hf_repository_factory=lambda _repo, _token: hf_repo,
    )

    with pytest.raises(HfProjectStateValidationError, match="signed-in validator"):
        router.save_validation("project-a", _validation(), expected_version=0)

    assert fallback.saved == []
    assert hf_repo.saved == []


def test_project_aware_repository_routes_bucket_write_with_validator_token() -> None:
    fallback = FakeValidationStateRepository("fallback")
    bucket_repo = FakeValidationStateRepository("bucket")
    received: list[tuple[str, str]] = []
    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="owner/project-a",
        validation_backend=HF_BUCKET_VALIDATION_BACKEND,
        validation_bucket_id="owner/project-a_validation_state",
    )
    router = ProjectAwareValidationRepository(
        fallback_repository=fallback,
        project_lookup=lambda slug: project if slug == "project-a" else None,
        token_provider=lambda _: "hf_admin_token",
        actor_token_provider=lambda _project, username: "hf_validator_token" if username == "validator-a" else None,
        enable_hf_bucket_validations=True,
        bucket_repository_factory=lambda bucket, token: (received.append((bucket, token)) or bucket_repo),
    )

    assert router.save_validation("project-a", _validation(), expected_version=0) == 1

    assert received == [("owner/project-a_validation_state", "hf_validator_token")]
    assert bucket_repo.saved == [("project-a", "audio-a-0000000001")]


def test_project_aware_repository_bucket_write_does_not_fall_back_to_admin_token() -> None:
    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="owner/project-a",
        validation_backend=HF_BUCKET_VALIDATION_BACKEND,
        validation_bucket_id="owner/project-a_validation_state",
    )
    router = ProjectAwareValidationRepository(
        fallback_repository=FakeValidationStateRepository("fallback"),
        project_lookup=lambda _: project,
        token_provider=lambda _: "hf_admin_token",
        actor_token_provider=lambda _project, _username: None,
        enable_hf_bucket_validations=True,
    )

    with pytest.raises(HfBucketValidationError, match="signed-in validator"):
        router.save_validation("project-a", _validation(), expected_version=0)


def test_project_aware_repository_never_falls_back_when_declared_bucket_is_disabled() -> None:
    fallback = FakeValidationStateRepository("fallback")
    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="owner/project-a",
        validation_backend=HF_BUCKET_VALIDATION_BACKEND,
        validation_bucket_id="owner/project-a_validation_state",
    )
    router = ProjectAwareValidationRepository(
        fallback_repository=fallback,
        project_lookup=lambda _: project,
        token_provider=lambda _: "hf_admin_token",
        actor_token_provider=lambda _project, _username: "hf_validator_token",
        enable_hf_bucket_validations=False,
    )

    with pytest.raises(HfBucketValidationError, match="Bucket validations are disabled"):
        router.save_validation("project-a", _validation(), expected_version=0)

    assert fallback.saved == []


def test_project_aware_repository_rejects_declared_bucket_without_bucket_id() -> None:
    fallback = FakeValidationStateRepository("fallback")
    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="owner/project-a",
        validation_backend=HF_BUCKET_VALIDATION_BACKEND,
    )
    router = ProjectAwareValidationRepository(
        fallback_repository=fallback,
        project_lookup=lambda _: project,
        token_provider=lambda _: "hf_admin_token",
        enable_hf_bucket_validations=True,
    )

    with pytest.raises(HfBucketValidationError, match="no validation bucket id"):
        router.load_current_snapshot("project-a")

    assert fallback.saved == []


def test_project_aware_repository_recent_events_falls_back_to_bounded_full_events() -> None:
    fallback = FakeValidationStateRepository("fallback")
    fallback.saved = [("project-a", "old"), ("project-a", "new")]
    router = ProjectAwareValidationRepository(
        fallback_repository=fallback,
        project_lookup=lambda _: None,
        token_provider=lambda _: None,
    )

    events = router.list_recent_events("project-a", limit=1)

    assert len(events) == 1
