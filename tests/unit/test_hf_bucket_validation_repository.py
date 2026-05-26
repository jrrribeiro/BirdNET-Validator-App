import json

import pytest

from src.domain.models import Validation
from src.repositories.append_only_validation_repository import OptimisticLockError
from src.repositories.hf_bucket_validation_repository import (
    HF_BUCKET_VALIDATION_BACKEND,
    HfBucketPermissionProbe,
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
        self.deletes: list[list[str]] = []

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

    def write_files(
        self,
        *,
        bucket_id: str,
        files: dict[str, bytes],
        token: str,
        delete_paths: list[str] | None = None,
    ) -> None:
        _ = (bucket_id, token)
        self.writes.append(dict(files))
        self.files.update({path: value.decode("utf-8") for path, value in files.items()})
        self.deletes.append(list(delete_paths or []))
        for path in delete_paths or []:
            self.files.pop(path, None)


class InaccessibleBucketApi(FakeBucketApi):
    def read_text(self, *, bucket_id: str, path_in_bucket: str, token: str) -> str:
        _ = (bucket_id, path_in_bucket, token)
        raise RuntimeError("404 Client Error: BucketNotFound")


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


def test_bucket_permission_probe_reads_writes_verifies_and_removes_marker() -> None:
    api = FakeBucketApi({"metadata/project.json": "{}"})
    probe = HfBucketPermissionProbe(api=api)

    result = probe.probe(
        bucket_id="owner/audio_validation_state",
        actor_username="validator-a",
        token="hf_validator",
    )

    assert result.bucket_id == "owner/audio_validation_state"
    assert result.actor_username == "validator-a"
    assert result.diagnostic_path not in api.files
    assert api.deletes[-1] == [result.diagnostic_path]


def test_save_validation_writes_event_and_snapshot_without_repo_commit() -> None:
    api = FakeBucketApi({"snapshots/current.json": json.dumps({"project_slug": "project-a", "items": {}})})
    repository = HfBucketValidationRepository(bucket_id="owner/audio_validation_state", token="hf_user", api=api)

    assert repository.save_validation("project-a", _validation(), expected_version=0) == 1

    assert len(api.writes) == 1
    paths = set(api.writes[0])
    assert "snapshots/current.json" in paths
    assert any(path.startswith("events/") for path in paths)
    assert repository.load_current_snapshot("project-a")["audio-a-0000000001"]["version"] == 1


def test_bucket_access_error_explains_that_app_invite_is_not_hub_permission() -> None:
    repository = HfBucketValidationRepository(
        bucket_id="owner/audio_validation_state",
        token="hf_user",
        api=InaccessibleBucketApi(),
    )

    with pytest.raises(HfBucketValidationError, match="invitation inside this app does not grant"):
        repository.load_current_snapshot("project-a")


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


def test_bucket_compaction_archives_active_events_and_preserves_audit_loading() -> None:
    api = FakeBucketApi(
        {
            "snapshots/current.json": json.dumps({"project_slug": "project-a", "items": {}}),
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
    repository = HfBucketValidationRepository(
        bucket_id="owner/audio_validation_state",
        token="hf_user",
        api=api,
        active_event_limit=2,
    )

    result = repository.compact_events("project-a")

    assert result.compacted_event_count == 2
    assert result.archive_path is not None
    assert not any(path.startswith("events/") for path in api.files)
    assert result.archive_path in api.files
    assert len(repository.list_events("project-a")) == 2
    assert set(repository.load_current_snapshot("project-a")) == {
        "audio-a-0000000001",
        "audio-b-0000000001",
    }
    api.files["snapshots/current.json"] = "{invalid json"
    assert set(repository.load_current_snapshot("project-a")) == {
        "audio-a-0000000001",
        "audio-b-0000000001",
    }


def test_save_validation_compacts_full_active_window_before_new_write() -> None:
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
    repository = HfBucketValidationRepository(
        bucket_id="owner/audio_validation_state",
        token="hf_user",
        api=api,
        active_event_limit=1,
    )

    repository.save_validation("project-a", _validation(), expected_version=0)

    archives = [path for path in api.files if path.startswith("archives/events/")]
    active = [path for path in api.files if path.startswith("events/")]
    assert len(archives) == 1
    assert len(active) == 1
    assert len(repository.list_events("project-a")) == 2


def test_bucket_events_deduplicate_same_event_present_in_parallel_archives() -> None:
    duplicate = {
        "event_id": "duplicate-event",
        "project_slug": "project-a",
        "detection_key": "audio-a-0000000001",
        "status": "positive",
        "validator": "validator-a",
        "timestamp": "2026-05-24T12:00:00+00:00",
        "new_version": 1,
    }
    api = FakeBucketApi(
        {
            "archives/events/a.jsonl": json.dumps(duplicate) + "\n",
            "archives/events/b.jsonl": json.dumps(duplicate) + "\n",
        }
    )
    repository = HfBucketValidationRepository(bucket_id="owner/audio_validation_state", token="hf_user", api=api)

    assert len(repository.list_events("project-a")) == 1


def test_bucket_recent_events_reads_only_needed_newest_archives() -> None:
    def event(event_id: str, timestamp: str) -> str:
        return json.dumps(
            {
                "event_id": event_id,
                "project_slug": "project-a",
                "detection_key": f"audio-{event_id}-0000000001",
                "status": "positive",
                "validator": "validator-a",
                "timestamp": timestamp,
                "new_version": 1,
            }
        ) + "\n"

    api = FakeBucketApi(
        {
            "archives/events/20260522.jsonl": event("old", "2026-05-22T12:00:00+00:00"),
            "archives/events/20260523.jsonl": event("middle", "2026-05-23T12:00:00+00:00"),
            "archives/events/20260524.jsonl": event("new", "2026-05-24T12:00:00+00:00"),
        }
    )
    repository = HfBucketValidationRepository(bucket_id="owner/audio_validation_state", token="hf_user", api=api)

    events = repository.list_recent_events("project-a", limit=2)

    assert [item["event_id"] for item in events] == ["new", "middle"]
    assert ["archives/events/20260522.jsonl"] not in api.read_batches


def test_bucket_compaction_refuses_to_delete_unreadable_active_event() -> None:
    api = FakeBucketApi({"events/20260524/broken.json": "{bad json"})
    repository = HfBucketValidationRepository(
        bucket_id="owner/audio_validation_state",
        token="hf_user",
        api=api,
        active_event_limit=1,
    )

    with pytest.raises(HfBucketValidationError, match="unreadable"):
        repository.compact_events("project-a")

    assert "events/20260524/broken.json" in api.files
    assert not api.deletes
