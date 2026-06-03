import json

import pytest

from src.domain.models import Validation
from src.repositories.append_only_validation_repository import OptimisticLockError
from src.repositories.hf_project_state_validation_repository import (
    HfProjectStateValidationError,
    HfProjectStateValidationRepository,
)


class FakeHfProjectStateFilesApi:
    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files = dict(files or {})
        self.commits: list[dict[str, object]] = []

    def list_repo_files(self, **kwargs):  # noqa: ANN001
        _ = kwargs
        return sorted(self.files)

    def read_text(self, **kwargs):  # noqa: ANN001
        path = str(kwargs["path_in_repo"])
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def create_commit(self, **kwargs):  # noqa: ANN001
        self.commits.append(dict(kwargs))
        for operation in kwargs["operations"]:
            payload = operation.path_or_fileobj
            if isinstance(payload, bytes):
                text = payload.decode("utf-8")
            else:
                text = str(payload)
            self.files[operation.path_in_repo] = text
        return object()


def _repo(api: FakeHfProjectStateFilesApi) -> HfProjectStateValidationRepository:
    return HfProjectStateValidationRepository(
        state_repo_id="jrrribeiro/upload_test2_state",
        token="hf_test",
        api=api,
    )


def _validation(status: str = "positive", detection_key: str = "audio-a-0000000001") -> Validation:
    return Validation(
        detection_key=detection_key,
        status=status,
        corrected_species=None,
        notes="reviewed",
        validator="validator-a",
    )


def test_save_validation_writes_event_and_current_snapshot() -> None:
    api = FakeHfProjectStateFilesApi(
        {
            "snapshots/current.json": json.dumps(
                {"schema_version": 1, "project_slug": "project-a", "items": {}}
            )
        }
    )
    repo = _repo(api)

    version = repo.save_validation("project-a", _validation(), expected_version=0)

    assert version == 1
    assert len(api.commits) == 1
    commit = api.commits[0]
    assert commit["repo_id"] == "jrrribeiro/upload_test2_state"
    assert commit["repo_type"] == "dataset"
    paths = [operation.path_in_repo for operation in commit["operations"]]
    assert paths[0].startswith("events/")
    assert paths[0].endswith(".json")
    assert paths[1] == "snapshots/current.json"
    snapshot = repo.load_current_snapshot("project-a")
    assert snapshot["audio-a-0000000001"]["status"] == "positive"
    assert snapshot["audio-a-0000000001"]["version"] == 1


def test_save_validation_raises_conflict_on_stale_expected_version() -> None:
    api = FakeHfProjectStateFilesApi()
    repo = _repo(api)

    assert repo.save_validation("project-a", _validation(), expected_version=0) == 1

    with pytest.raises(OptimisticLockError) as exc:
        repo.save_validation("project-a", _validation(status="negative"), expected_version=0)

    assert exc.value.current_version == 1


def test_load_current_snapshot_recovers_from_events_when_snapshot_missing() -> None:
    api = FakeHfProjectStateFilesApi(
        {
            "events/20260524/event-a.json": json.dumps(
                {
                    "project_slug": "project-a",
                    "detection_key": "audio-a",
                    "status": "positive",
                    "validator": "validator-a",
                    "timestamp": "2026-05-24T12:00:00+00:00",
                    "new_version": 1,
                }
            ),
            "events/.gitkeep": "",
        }
    )
    repo = _repo(api)

    snapshot = repo.load_current_snapshot("project-a")

    assert snapshot == {
        "audio-a": {
            "status": "positive",
            "corrected_species": None,
            "notes": "",
            "validator": "validator-a",
            "updated_at": "2026-05-24T12:00:00+00:00",
            "version": 1,
        }
    }


def test_list_events_sorts_and_filters_by_project() -> None:
    api = FakeHfProjectStateFilesApi(
        {
            "events/20260524/b.json": json.dumps(
                {
                    "project_slug": "project-a",
                    "detection_key": "audio-b",
                    "timestamp": "2026-05-24T12:01:00+00:00",
                }
            ),
            "events/20260524/a.json": json.dumps(
                {
                    "project_slug": "project-a",
                    "detection_key": "audio-a",
                    "timestamp": "2026-05-24T12:00:00+00:00",
                }
            ),
            "events/20260524/other.json": json.dumps(
                {
                    "project_slug": "project-b",
                    "detection_key": "audio-x",
                    "timestamp": "2026-05-24T11:00:00+00:00",
                }
            ),
        }
    )
    repo = _repo(api)

    events = repo.list_events("project-a")

    assert [event["detection_key"] for event in events] == ["audio-a", "audio-b"]


def test_repository_requires_state_repo_and_token() -> None:
    api = FakeHfProjectStateFilesApi()

    with pytest.raises(HfProjectStateValidationError):
        HfProjectStateValidationRepository(state_repo_id="", token="hf_test", api=api)
    with pytest.raises(HfProjectStateValidationError):
        HfProjectStateValidationRepository(state_repo_id="owner/repo_state", token="", api=api)
