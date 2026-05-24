import json

import pytest

from src.domain.models import Validation
from src.repositories.append_only_validation_repository import OptimisticLockError
from src.repositories.hf_bucket_validation_repository import (
    HF_BUCKET_VALIDATION_BACKEND,
    HfBucketValidationError,
    HfBucketValidationInitializer,
    HfBucketValidationRepository,
    default_validation_bucket_id,
)


class FakeBucketApi:
    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.files = dict(files or {})
        self.created: list[tuple[str, str]] = []
        self.read_batches: list[list[str]] = []
        self.writes: list[dict[str, bytes]] = []

    def create_private_bucket(self, *, bucket_id: str, token: str) -> None:
        self.created.append((bucket_id, token))

    def list_files(self, *, bucket_id: str, prefix: str, token: str) -> list[str]:
        _ = (bucket_id, token)
        return sorted(path for path in self.files if path.startswith(prefix))

    def read_text(self, *, bucket_id: str, path_in_bucket: str, token: str) -> str:
        _ = (bucket_id, token)
        if path_in_bucket not in self.files:
            raise FileNotFoundError(path_in_bucket)
        return self.files[path_in_bucket]

    def read_texts(self, *, bucket_id: str, paths_in_bucket: list[str], token: str) -> dict[str, str]:
        _ = (bucket_id, token)
        self.read_batches.append(list(paths_in_bucket))
        return {path: self.files[path] for path in paths_in_bucket if path in self.files}

    def write_files(self, *, bucket_id: str, files: dict[str, bytes], token: str) -> None:
        _ = (bucket_id, token)
        self.writes.append(dict(files))
        self.files.update({path: value.decode("utf-8") for path, value in files.items()})


def _validation(status: str = "positive") -> Validation:
    return Validation(
        detection_key="audio-a-0000000001",
        status=status,
        corrected_species=None,
        notes="reviewed",
        validator="validator-a",
    )


def test_default_validation_bucket_id_uses_project_namespace() -> None:
    assert default_validation_bucket_id("owner/audio") == "owner/audio_validation_state"


def test_initializer_creates_private_bucket_manifest_and_snapshot() -> None:
    api = FakeBucketApi()
    initializer = HfBucketValidationInitializer(api=api)

    result = initializer.initialize(project_slug="project-a", dataset_repo_id="owner/audio", token="hf_admin")

    assert result.bucket_id == "owner/audio_validation_state"
    assert result.initialized is True
    assert api.created == [("owner/audio_validation_state", "hf_admin")]
    payload = json.loads(api.files["metadata/project.json"])
    assert payload["validation_backend"] == HF_BUCKET_VALIDATION_BACKEND
    assert "snapshots/current.json" in api.files


def test_initializer_reuses_manifest_but_refuses_unknown_existing_data() -> None:
    reused = HfBucketValidationInitializer(api=FakeBucketApi({"metadata/project.json": "{}"}))
    result = reused.initialize(project_slug="project-a", dataset_repo_id="owner/audio", token="hf_admin")
    assert result.reused_existing is True

    unsafe = HfBucketValidationInitializer(api=FakeBucketApi({"other.txt": "data"}))
    with pytest.raises(HfBucketValidationError, match="refusing automatic initialization"):
        unsafe.initialize(project_slug="project-a", dataset_repo_id="owner/audio", token="hf_admin")


def test_save_validation_writes_event_and_snapshot_without_repo_commit() -> None:
    api = FakeBucketApi({"snapshots/current.json": json.dumps({"project_slug": "project-a", "items": {}})})
    repository = HfBucketValidationRepository(bucket_id="owner/audio_validation_state", token="hf_user", api=api)

    assert repository.save_validation("project-a", _validation(), expected_version=0) == 1

    assert len(api.writes) == 1
    paths = set(api.writes[0])
    assert "snapshots/current.json" in paths
    assert any(path.startswith("events/") for path in paths)
    assert repository.load_current_snapshot("project-a")["audio-a-0000000001"]["version"] == 1


def test_bucket_snapshot_recovers_from_events_and_rejects_stale_version() -> None:
    api = FakeBucketApi(
        {
            "events/20260524/a.json": json.dumps(
                {
                    "project_slug": "project-a",
                    "detection_key": "audio-a-0000000001",
                    "status": "positive",
                    "validator": "validator-a",
                    "timestamp": "2026-05-24T12:00:00+00:00",
                    "new_version": 1,
                }
            )
        }
    )
    repository = HfBucketValidationRepository(bucket_id="owner/audio_validation_state", token="hf_user", api=api)

    assert repository.load_current_snapshot("project-a")["audio-a-0000000001"]["status"] == "positive"
    with pytest.raises(OptimisticLockError):
        repository.save_validation("project-a", _validation(status="negative"), expected_version=0)


def test_bucket_snapshot_reconciles_event_missing_after_parallel_snapshot_overwrite() -> None:
    api = FakeBucketApi(
        {
            "snapshots/current.json": json.dumps(
                {
                    "project_slug": "project-a",
                    "items": {
                        "audio-b-0000000001": {
                            "status": "negative",
                            "validator": "validator-b",
                            "version": 1,
                        }
                    },
                }
            ),
            "events/20260524/a.json": json.dumps(
                {
                    "event_id": "event-a",
                    "project_slug": "project-a",
                    "detection_key": "audio-a-0000000001",
                    "status": "positive",
                    "validator": "validator-a",
                    "timestamp": "2026-05-24T12:00:00+00:00",
                    "new_version": 1,
                }
            ),
            "events/20260524/b.json": json.dumps(
                {
                    "event_id": "event-b",
                    "project_slug": "project-a",
                    "detection_key": "audio-b-0000000001",
                    "status": "negative",
                    "validator": "validator-b",
                    "timestamp": "2026-05-24T12:00:01+00:00",
                    "new_version": 1,
                }
            ),
        }
    )
    repository = HfBucketValidationRepository(bucket_id="owner/audio_validation_state", token="hf_user", api=api)

    snapshot = repository.load_current_snapshot("project-a")

    assert set(snapshot) == {"audio-a-0000000001", "audio-b-0000000001"}
    assert snapshot["audio-a-0000000001"]["status"] == "positive"
    assert api.read_batches == [["events/20260524/a.json", "events/20260524/b.json"]]


def test_bucket_snapshot_marks_parallel_same_version_decisions_as_conflict() -> None:
    api = FakeBucketApi(
        {
            "events/20260524/a.json": json.dumps(
                {
                    "event_id": "event-a",
                    "project_slug": "project-a",
                    "detection_key": "audio-a-0000000001",
                    "status": "positive",
                    "validator": "validator-a",
                    "timestamp": "2026-05-24T12:00:00+00:00",
                    "new_version": 1,
                }
            ),
            "events/20260524/b.json": json.dumps(
                {
                    "event_id": "event-b",
                    "project_slug": "project-a",
                    "detection_key": "audio-a-0000000001",
                    "status": "negative",
                    "validator": "validator-b",
                    "timestamp": "2026-05-24T12:00:01+00:00",
                    "new_version": 1,
                }
            ),
        }
    )
    repository = HfBucketValidationRepository(bucket_id="owner/audio_validation_state", token="hf_user", api=api)

    state = repository.load_current_snapshot("project-a")["audio-a-0000000001"]

    assert state["status"] == "negative"
    assert state["conflict"] is True
    assert state["conflict_reason"] == "parallel_events_same_version"
