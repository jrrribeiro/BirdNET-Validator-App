from dataclasses import dataclass
from datetime import date
from pathlib import Path
import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.ui.app_factory import (
    _build_validation_export_rows,
    _build_validation_report,
    _build_species_status_map,
    _cleanup_selected_audio,
    _advance_to_next_row_with_title,
    _autofetch_first_row,
    _extract_audio_id,
    _extract_detection_key,
    _find_detection_row_index,
    _fetch_selected_audio,
    _page_to_table,
    _post_validation_queue_anchor,
    _save_selected_validation,
    _save_selected_validation_with_refresh,
    _advance_after_validation_with_title,
    _selected_dataframe_row_index,
    _selected_segment_card,
    _reapply_last_conflict_validation_with_refresh,
    _batch_validate_conflicts,
    create_app,
    _load_seed_detections,
    _validate_seed_file,
    _build_detection_repository,
    _get_project_detection_count,
    _build_queue_badge,
    _build_validation_summary_cards_for_species,
    _load_projects_from_file,
    _load_user_access_from_file,
    _bootstrap_auth_and_projects,
    _discover_hf_admin_state_repos,
    _persist_bootstrap_state,
    _resolve_username_login_policy,
    _resolve_project_fetch_token,
    _initialize_hf_admin_storage,
    _write_validation_export,
    _validation_shortcuts_script,
    _species_status_dropdown_script,
    _species_status_payload,
    _species_dropdown_choices,
    _corrected_species_error_payload,
    _corrected_species_required_script,
    _corrected_species_ui_after_validation,
    VALIDATION_REJECT_CORRECTION_REQUIRED_STATUS,
)
from src.auth.auth_service import AuthService
from src.config.runtime_config import RuntimeConfig
from src.domain.models import Detection, Project, Role
from src.repositories.hf_bucket_validation_repository import HfBucketValidationError, HfBucketValidationInitResult
from src.repositories.state_safety import StateSafetyError
from src.services.hf_project_state_store import HfProjectStateStoreError, HfProjectStateStoreLoadedProject
from src.ui.admin_panel import AdminPanelManager


@dataclass
class FakeFetchResult:
    cache_key: str
    local_path: str
    source: str


class FakeAudioService:
    def __init__(self) -> None:
        self.cleaned: list[str] = []

    def fetch(self, dataset_repo: str, audio_id: str) -> FakeFetchResult:
        _ = dataset_repo
        return FakeFetchResult(cache_key=f"key:{audio_id}", local_path=f"/tmp/{audio_id}.wav", source="remote")

    def cleanup_after_validation(self, cache_key: str) -> None:
        self.cleaned.append(cache_key)


class FakeDatasetInfoApi:
    def __init__(self, private: bool, owner_username: str = "jrrribeiro") -> None:
        self.private = private
        self.owner_username = owner_username

    def whoami(self, token: str):  # noqa: ANN001
        assert token == "hf_admin"
        return {"name": self.owner_username}

    def repo_info(self, *, repo_id: str, repo_type: str, token: str):  # noqa: ANN001
        assert repo_id == "jrrribeiro/private-audio"
        assert repo_type == "dataset"
        assert token == "hf_admin"
        return SimpleNamespace(private=self.private)


class FakeBucketInitializer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str | None]] = []

    def initialize(self, *, project_slug: str, dataset_repo_id: str, token: str | None):
        self.calls.append((project_slug, dataset_repo_id, token))
        return HfBucketValidationInitResult(
            bucket_id="jrrribeiro/private-audio_validation_state",
            initialized=True,
            reused_existing=False,
        )


class FakeStateDiscoveryApi:
    def whoami(self, token: str):  # noqa: ANN001
        assert token == "hf_storage"
        return {"name": "jrrribeiro"}

    def list_datasets(self, *, author: str, token: str):  # noqa: ANN001
        assert author == "jrrribeiro"
        assert token == "hf_storage"
        return [
            SimpleNamespace(id="jrrribeiro/project-a_state"),
            SimpleNamespace(id="jrrribeiro/audio-source"),
            SimpleNamespace(id="someone-else/project-b_state"),
        ]


class InaccessibleAudioService:
    def fetch(self, dataset_repo: str, audio_id: str, **kwargs: object) -> FakeFetchResult:
        _ = (dataset_repo, audio_id, kwargs)
        raise RuntimeError("404 Client Error: Repository Not Found")


class FakeValidationService:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def validate_detection(
        self,
        project_slug: str,
        detection_key: str,
        status: str,
        validator: str,
        notes: str = "",
        corrected_species: str | None = None,
        expected_version: int | None = None,
    ) -> dict[str, str]:
        payload = {
            "project_slug": project_slug,
            "detection_key": detection_key,
            "status": status,
            "validator": validator,
            "notes": notes,
            "corrected_species": corrected_species or "",
            "expected_version": str(expected_version),
        }
        self.calls.append(payload)
        return payload


class NoopInviteNotifier:
    def send(self, payload):  # noqa: ANN001
        _ = payload
        return True, "not sent"


class FakeConflictValidationService:
    def validate_detection(
        self,
        project_slug: str,
        detection_key: str,
        status: str,
        validator: str,
        notes: str = "",
        corrected_species: str | None = None,
        expected_version: int | None = None,
    ) -> dict[str, str]:
        _ = project_slug
        _ = detection_key
        _ = status
        _ = validator
        _ = notes
        _ = corrected_species
        _ = expected_version
        from src.repositories.append_only_validation_repository import OptimisticLockError

        raise OptimisticLockError("dkey_01", expected_version or 0, 3)


class FakeSnapshotReader:
    def __init__(self) -> None:
        self.snapshot: dict[str, dict[str, object]] = {
            "dkey_01": {
                "status": "positive",
                "validator": "validator-demo",
                "updated_at": "2026-03-25T10:00:00+00:00",
                "version": 2,
            },
            "dkey_02": {
                "status": "negative",
                "validator": "validator-other",
                "updated_at": "2026-03-20T10:00:00+00:00",
                "version": 1,
            }
        }
        self.events: list[dict[str, object]] = [
            {"detection_key": "dkey_01", "status": "positive"},
            {"detection_key": "dkey_02", "status": "negative"},
        ]

    def load_current_snapshot(self, project_slug: str, actor_username: str = "") -> dict[str, dict[str, object]]:
        _ = (project_slug, actor_username)
        return self.snapshot

    def list_events(self, project_slug: str, actor_username: str = "") -> list[dict[str, object]]:
        _ = (project_slug, actor_username)
        return self.events


class FakeQueueService:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] = {}

    class _Page:
        def __init__(self) -> None:
            self.page = 1
            self.total_pages = 1
            self.total_items = 2
            self.items = [
                type(
                    "DetectionLike",
                    (),
                    {
                        "detection_key": "dkey_01",
                        "audio_id": "audio_01",
                        "scientific_name": "sp",
                        "confidence": 0.9,
                        "start_time": 0.0,
                        "end_time": 1.0,
                    },
                )(),
                type(
                    "DetectionLike",
                    (),
                    {
                        "detection_key": "dkey_02",
                        "audio_id": "audio_02",
                        "scientific_name": "sp2",
                        "confidence": 0.85,
                        "start_time": 1.0,
                        "end_time": 2.0,
                    },
                )(),
            ]

    def get_page(self, **kwargs: object) -> "FakeQueueService._Page":
        self.last_kwargs = kwargs
        return FakeQueueService._Page()


def test_build_validation_summary_cards_for_species_uses_species_total_not_page_rows() -> None:
    class QueueWithListAll(FakeQueueService):
        def list_all_detections(self, **kwargs: object) -> list[object]:
            self.last_kwargs = kwargs
            return [
                SimpleNamespace(
                    detection_key="dkey_01",
                    audio_id="audio_01",
                    scientific_name="sp",
                    confidence=0.9,
                    start_time=0,
                    end_time=1,
                ),
                SimpleNamespace(
                    detection_key="dkey_02",
                    audio_id="audio_02",
                    scientific_name="sp",
                    confidence=0.8,
                    start_time=1,
                    end_time=2,
                ),
                SimpleNamespace(
                    detection_key="dkey_03",
                    audio_id="audio_03",
                    scientific_name="sp",
                    confidence=0.7,
                    start_time=2,
                    end_time=3,
                ),
            ]

    html = _build_validation_summary_cards_for_species(
        queue_service=QueueWithListAll(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="project-a",
        scientific_name="sp",
        min_confidence=0.0,
        actor_username="validator",
    )

    assert "Queue total" in html
    assert "segments in species" in html
    assert "66.7%" in html
    assert "1 pending in species" in html


def test_initialize_hf_admin_storage_sets_bucket_without_persisting_shared_token() -> None:
    project = Project(
        project_slug="private-audio",
        name="Private Audio",
        dataset_repo_id="jrrribeiro/private-audio",
        dataset_token="hf_shared_should_be_removed",
    )
    bucket_initializer = FakeBucketInitializer()

    result = _initialize_hf_admin_storage(
        project=project,
        token="hf_admin",
        api=FakeDatasetInfoApi(private=True),  # type: ignore[arg-type]
        bucket_initializer=bucket_initializer,  # type: ignore[arg-type]
    )

    assert result.bucket_id == "jrrribeiro/private-audio_validation_state"
    assert project.validation_bucket_id == result.bucket_id
    assert project.validation_backend == "hf_bucket"
    assert project.dataset_token is None
    assert bucket_initializer.calls == [("private-audio", "jrrribeiro/private-audio", "hf_admin")]


def test_initialize_hf_admin_storage_rejects_public_source_dataset() -> None:
    project = Project(
        project_slug="private-audio",
        name="Private Audio",
        dataset_repo_id="jrrribeiro/private-audio",
    )
    bucket_initializer = FakeBucketInitializer()

    with pytest.raises(HfBucketValidationError, match="only accepts a private dataset"):
        _initialize_hf_admin_storage(
            project=project,
            token="hf_admin",
            api=FakeDatasetInfoApi(private=False),  # type: ignore[arg-type]
            bucket_initializer=bucket_initializer,  # type: ignore[arg-type]
        )

    assert bucket_initializer.calls == []


def test_initialize_hf_admin_storage_requires_storage_owner_personal_namespace() -> None:
    project = Project(
        project_slug="private-audio",
        name="Private Audio",
        dataset_repo_id="jrrribeiro/private-audio",
    )
    bucket_initializer = FakeBucketInitializer()

    with pytest.raises(HfBucketValidationError, match="personal namespace"):
        _initialize_hf_admin_storage(
            project=project,
            token="hf_admin",
            api=FakeDatasetInfoApi(private=True, owner_username="another-admin"),  # type: ignore[arg-type]
            bucket_initializer=bucket_initializer,  # type: ignore[arg-type]
        )

    assert bucket_initializer.calls == []


def test_initialize_hf_admin_storage_requires_collaborative_app_visibility() -> None:
    project = Project(
        project_slug="private-audio",
        name="Private Audio",
        dataset_repo_id="jrrribeiro/private-audio",
        visibility="private",
    )
    bucket_initializer = FakeBucketInitializer()

    with pytest.raises(HfBucketValidationError, match="visibility 'collaborative'"):
        _initialize_hf_admin_storage(
            project=project,
            token="hf_admin",
            api=FakeDatasetInfoApi(private=True),  # type: ignore[arg-type]
            bucket_initializer=bucket_initializer,  # type: ignore[arg-type]
        )

    assert bucket_initializer.calls == []


def test_discover_hf_admin_state_repos_filters_personal_companion_datasets() -> None:
    repo_ids, warning = _discover_hf_admin_state_repos(
        token="hf_storage",
        api=FakeStateDiscoveryApi(),  # type: ignore[arg-type]
    )

    assert warning == ""
    assert repo_ids == ("jrrribeiro/project-a_state",)


def test_extract_audio_id_from_list_rows() -> None:
    rows = [["k1", "audio_01", "sp", 0.9, 0.0, 1.0]]
    assert _extract_audio_id(rows, 0) == "audio_01"


def test_extract_audio_id_from_dataframe_rows() -> None:
    frame = pd.DataFrame([["k1", "audio_02", "sp", 0.9, 0.0, 1.0]])
    assert _extract_audio_id(frame, 0) == "audio_02"


def test_fetch_selected_audio_success() -> None:
    service = FakeAudioService()
    rows = [["k1", "audio_03", "sp", 0.9, 0.0, 1.0]]

    path, cache_key, status = _fetch_selected_audio(
        audio_service=service,
        dataset_repo="org/dataset",
        rows=rows,
        selected_index=0,
        previous_cache_key="",
    )

    assert path == "/tmp/audio_03.wav"
    assert cache_key == "key:audio_03"
    assert "Audio loaded" in status


def test_extract_detection_key_from_rows() -> None:
    rows = [["dkey_01", "audio_01", "sp", 0.9, 0.0, 1.0]]
    assert _extract_detection_key(rows, 0) == "dkey_01"


def test_fetch_selected_audio_validates_repo() -> None:
    service = FakeAudioService()
    rows = [["k1", "audio_03", "sp", 0.9, 0.0, 1.0]]

    path, cache_key, status = _fetch_selected_audio(
        audio_service=service,
        dataset_repo="   ",
        rows=rows,
        selected_index=0,
        previous_cache_key="old-key",
    )

    assert path is None
    assert cache_key == ""
    assert "Provide dataset repo" in status


def test_fetch_selected_audio_explains_private_dataset_access_uses_backend_credential() -> None:
    path, cache_key, status = _fetch_selected_audio(
        audio_service=InaccessibleAudioService(),
        dataset_repo="owner/private-audio",
        rows=[["k1", "audio_03", "sp", 0.9, 0.0, 1.0]],
        selected_index=0,
        previous_cache_key="",
        hf_token="hf_validator",
    )

    assert path is None
    assert cache_key == ""
    assert "credential used by the app does not have read access" in status
    assert "BIRDNET_HF_STORAGE_TOKEN" in status


def test_fetch_selected_audio_explains_missing_private_dataset_credential() -> None:
    _, _, status = _fetch_selected_audio(
        audio_service=InaccessibleAudioService(),
        dataset_repo="owner/private-audio",
        rows=[["k1", "audio_03", "sp", 0.9, 0.0, 1.0]],
        selected_index=0,
        previous_cache_key="old",
    )

    assert "No credential with private dataset read access was provided" in status


def test_cleanup_selected_audio() -> None:
    service = FakeAudioService()

    status, player_value = _cleanup_selected_audio(service, "key:audio_03")

    assert "Audio cache cleaned" in status
    assert player_value is None
    assert service.cleaned == ["key:audio_03"]


def test_save_selected_validation_saves_and_cleans_audio_cache() -> None:
    audio_service = FakeAudioService()
    validation_service = FakeValidationService()
    rows = [["0000000000001111", "audio_11", "sp", 0.9, 0.0, 1.0, "pending", 0]]

    status, cache_key, audio_path = _save_selected_validation(
        validation_service=validation_service,
        audio_service=audio_service,
        project_slug="demo-project",
        rows=rows,
        selected_index=0,
        status_value="positive",
        validator="validator-demo",
        notes="ok",
        cache_key="cache:audio_11",
    )

    assert "Validation saved" in status
    assert cache_key == ""
    assert audio_path is None
    assert len(validation_service.calls) == 1
    assert validation_service.calls[0]["detection_key"] == "0000000000001111"
    assert validation_service.calls[0]["expected_version"] == "0"
    assert audio_service.cleaned == ["cache:audio_11"]


def test_save_selected_validation_returns_conflict_message() -> None:
    audio_service = FakeAudioService()
    validation_service = FakeConflictValidationService()
    rows = [["0000000000001111", "audio_11", "sp", 0.9, 0.0, 1.0, "pending", 0]]

    status, cache_key, audio_path = _save_selected_validation(
        validation_service=validation_service,
        audio_service=audio_service,
        project_slug="demo-project",
        rows=rows,
        selected_index=0,
        status_value="positive",
        validator="validator-demo",
        notes="ok",
        cache_key="cache:audio_11",
    )

    assert "Concurrency conflict" in status
    assert cache_key == "cache:audio_11"
    assert audio_path is None


def test_build_validation_report() -> None:
    report = _build_validation_report(FakeSnapshotReader(), "demo-project")

    assert "Project: demo-project" in report
    assert "Append-only events: 2" in report
    assert "Detections with current state: 2" in report
    assert "positive=1" in report
    assert "negative=1" in report


def test_build_validation_export_rows_keep_detections_metadata_and_current_validation() -> None:
    columns, rows = _build_validation_export_rows(
        [
            Detection(
                detection_key="0000000000000002",
                audio_id="Species B/second.wav",
                scientific_name="Species B",
                confidence=0.92,
                start_time=6.0,
                end_time=9.0,
                source_metadata={"Common Name": "Bird B", "Latitude": -3.1},
            ),
            Detection(
                detection_key="0000000000000001",
                audio_id="Species A/first.wav",
                scientific_name="Species A",
                confidence=0.87,
                start_time=0.0,
                end_time=3.0,
                source_metadata={"Common Name": "Bird A", "Latitude": -2.9},
            ),
        ],
        {
            "0000000000000001": {
                "status": "negative",
                "corrected_species": "Corrected species",
                "notes": "Checked twice",
                "validator": "scientist",
                "updated_at": "2026-05-22T12:00:00+00:00",
                "version": 2,
                "conflict": True,
                "conflict_reason": "parallel_events_same_version",
            }
        },
        project_slug="analysis-project",
        dataset_repo_id="birds/analysis-project",
    )

    assert columns[:8] == [
        "project_slug",
        "dataset_repo_id",
        "detection_key",
        "audio_id",
        "detection_scientific_name",
        "detection_confidence",
        "detection_start_time",
        "detection_end_time",
    ]
    assert columns[8:10] == ["source_Common Name", "source_Latitude"]
    assert rows[0]["detection_key"] == "0000000000000001"
    assert rows[0]["validation_status"] == "negative"
    assert rows[0]["validation_corrected_species"] == "Corrected species"
    assert rows[0]["validation_effective_species"] == "Corrected species"
    assert rows[0]["validation_version"] == 2
    assert rows[0]["validation_conflict"] is True
    assert rows[0]["validation_conflict_reason"] == "parallel_events_same_version"
    assert rows[1]["validation_status"] == "pending"
    assert rows[1]["validation_effective_species"] == "Species B"
    assert rows[1]["validation_reviewed"] is False


def test_write_validation_export_creates_csv_and_xlsx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.ui.app_factory.tempfile.mkdtemp", lambda prefix: str(tmp_path))
    detections = [
        Detection(
            detection_key="0000000000000001",
            audio_id="Species A/first.wav",
            scientific_name="Species A",
            confidence=0.87,
            start_time=0.0,
            end_time=3.0,
            source_metadata={"source_file": "first.wav"},
        )
    ]
    snapshot = {"0000000000000001": {"status": "negative", "validator": "reviewer", "version": 1}}

    csv_path = _write_validation_export(
        detections,
        snapshot,
        project_slug="analysis-project",
        dataset_repo_id="birds/analysis-project",
        file_format="csv",
    )
    xlsx_path = _write_validation_export(
        detections,
        snapshot,
        project_slug="analysis-project",
        dataset_repo_id="birds/analysis-project",
        file_format="xlsx",
    )

    csv_frame = pd.read_csv(csv_path)
    xlsx_frame = pd.read_excel(xlsx_path)
    assert csv_frame.loc[0, "validation_status"] == "negative"
    assert csv_frame.loc[0, "source_source_file"] == "first.wav"
    assert xlsx_frame.loc[0, "validation_validator"] == "reviewer"
    assert xlsx_frame.loc[0, "dataset_repo_id"] == "birds/analysis-project"


def test_write_validation_export_splits_large_xlsx_into_sheets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.ui.app_factory.tempfile.mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr("src.ui.app_factory._XLSX_EXPORT_MAX_DATA_ROWS", 1)
    detections = [
        Detection(
            detection_key=f"000000000000000{index}",
            audio_id=f"Species A/{index}.wav",
            scientific_name="Species A",
            confidence=0.87,
            start_time=0.0,
            end_time=3.0,
        )
        for index in (1, 2)
    ]

    path = _write_validation_export(
        detections,
        {},
        project_slug="large-analysis-project",
        dataset_repo_id="birds/large-analysis-project",
        file_format="xlsx",
    )

    assert pd.ExcelFile(path).sheet_names == ["validation_data", "validation_data_2"]


def test_page_to_table_includes_validation_status() -> None:
    queue = FakeQueueService()
    rows, status, page = _page_to_table(
        service=queue,
        snapshot_reader=FakeSnapshotReader(),
        project_slug="kenya-2024",
        page=1,
        scientific_name="",
        min_confidence=0.0,
    )

    assert page == 1
    assert "Page 1/1" in status
    assert rows[0][0] == "dkey_01"
    assert rows[0][6] == "positive"
    assert rows[0][7] == 2
    assert rows[0][8] == ""
    assert rows[0][9] == ""
    assert queue.last_kwargs["project_slug"] == "kenya-2024"


def test_page_to_table_prioritizes_pending_then_confidence() -> None:
    class QueueWithMixedStatus:
        def list_all_detections(self, **kwargs: object) -> list[object]:
            _ = kwargs
            return [
                SimpleNamespace(
                    detection_key="reviewed_high",
                    audio_id="reviewed_high.wav",
                    scientific_name="sp",
                    confidence=0.99,
                    start_time=0,
                    end_time=1,
                ),
                SimpleNamespace(
                    detection_key="pending_mid",
                    audio_id="pending_mid.wav",
                    scientific_name="sp",
                    confidence=0.82,
                    start_time=1,
                    end_time=2,
                ),
                SimpleNamespace(
                    detection_key="pending_low",
                    audio_id="pending_low.wav",
                    scientific_name="sp",
                    confidence=0.61,
                    start_time=2,
                    end_time=3,
                ),
                SimpleNamespace(
                    detection_key="reviewed_low",
                    audio_id="reviewed_low.wav",
                    scientific_name="sp",
                    confidence=0.55,
                    start_time=3,
                    end_time=4,
                ),
            ]

    class SnapshotWithMixedStatus:
        def load_current_snapshot(self, project_slug: str, actor_username: str = "") -> dict[str, dict[str, object]]:
            _ = (project_slug, actor_username)
            return {
                "reviewed_high": {"status": "positive", "version": 1},
                "reviewed_low": {"status": "negative", "version": 1},
            }

    rows, _, _ = _page_to_table(
        service=QueueWithMixedStatus(),
        snapshot_reader=SnapshotWithMixedStatus(),
        project_slug="demo-project",
        page=1,
        scientific_name="sp",
        min_confidence=0.0,
        page_size=10,
    )

    assert [row[0] for row in rows] == [
        "pending_mid",
        "pending_low",
        "reviewed_high",
        "reviewed_low",
    ]


def test_build_species_status_map_marks_progress_levels() -> None:
    class QueueWithSpecies:
        def list_all_detections(self, **kwargs: object) -> list[object]:
            _ = kwargs
            return [
                SimpleNamespace(detection_key="sp_a_1", scientific_name="Species A"),
                SimpleNamespace(detection_key="sp_a_2", scientific_name="Species A"),
                SimpleNamespace(detection_key="sp_b_1", scientific_name="Species B"),
                SimpleNamespace(detection_key="sp_c_1", scientific_name="Species C"),
            ]

    class SnapshotWithSpecies:
        def load_current_snapshot(self, project_slug: str, actor_username: str = "") -> dict[str, dict[str, object]]:
            _ = (project_slug, actor_username)
            return {
                "sp_a_1": {"status": "positive"},
                "sp_b_1": {"status": "negative"},
            }

    status_map = _build_species_status_map(
        queue_service=QueueWithSpecies(),
        snapshot_reader=SnapshotWithSpecies(),
        project_slug="demo-project",
        page_size=10,
    )

    assert status_map["Species A"] == {"status": "partial", "total": 2, "reviewed": 1}
    assert status_map["Species B"] == {"status": "complete", "total": 1, "reviewed": 1}
    assert status_map["Species C"] == {"status": "unvalidated", "total": 1, "reviewed": 0}


def test_species_status_dropdown_script_targets_dropdown_without_changing_choices() -> None:
    script = _species_status_dropdown_script()

    assert "bn-species-filter" in script
    assert "bn-species-status-payload" in script
    assert "bn-species-option-complete" in script
    assert "bn-species-option-partial" in script
    assert "bn-species-option-unvalidated" in script


def test_species_status_payload_is_hidden_json_carrier() -> None:
    payload = _species_status_payload({"Species A": {"status": "complete", "total": 1, "reviewed": 1}})

    assert 'class="bn-species-status-data"' in payload
    assert "data-json=" in payload
    assert "&quot;Species A&quot;" in payload
    assert "&quot;complete&quot;" in payload


def test_species_dropdown_choices_keep_clean_backend_values() -> None:
    choices = _species_dropdown_choices(
        {
            "Species A": {"status": "partial", "total": 2, "reviewed": 1},
            "Species B": {"status": "complete", "total": 1, "reviewed": 1},
            "Species C": {"status": "unvalidated", "total": 1, "reviewed": 0},
        }
    )

    assert choices == [
        ("🟡 Species A", "Species A"),
        ("🟢 Species B", "Species B"),
        ("⚪ Species C", "Species C"),
    ]


def test_page_to_table_marks_conflict_row() -> None:
    rows, _, _ = _page_to_table(
        service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        conflict_detection_key="dkey_01",
    )

    assert rows[0][8] == "CONFLICT"
    assert rows[0][9] == "HIGH"


def test_page_to_table_marks_persisted_backend_conflict_row() -> None:
    reader = FakeSnapshotReader()
    reader.snapshot["dkey_01"]["conflict"] = True
    rows, _, _ = _page_to_table(
        service=FakeQueueService(),
        snapshot_reader=reader,
        project_slug="demo-project",
        page=1,
        scientific_name="",
        min_confidence=0.0,
    )

    assert rows[0][8] == "CONFLICT"
    assert rows[0][9] == "HIGH"


def test_page_to_table_conflicts_only_filter_hides_non_conflicts() -> None:
    rows, status, _ = _page_to_table(
        service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        show_conflicts_only=True,
    )

    assert rows == []
    assert "Conflicts only: 0 item(ns)" in status


def test_page_to_table_conflicts_only_filter_keeps_conflict_rows() -> None:
    rows, status, _ = _page_to_table(
        service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        conflict_detection_key="dkey_01",
        show_conflicts_only=True,
    )

    assert len(rows) == 1
    assert rows[0][8] == "CONFLICT"
    assert "Conflicts only: 1 item(ns)" in status


def test_page_to_table_filters_by_validator() -> None:
    rows, _, _ = _page_to_table(
        service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        validator_filter="other",
    )

    assert len(rows) == 1
    assert rows[0][0] == "dkey_02"


def test_page_to_table_filters_by_status() -> None:
    rows, _, _ = _page_to_table(
        service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        status_filter="negative",
    )

    assert len(rows) == 1
    assert rows[0][0] == "dkey_02"


def test_page_to_table_filters_by_updated_after() -> None:
    rows, _, _ = _page_to_table(
        service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        updated_after="2026-03-24",
    )

    assert len(rows) == 1
    assert rows[0][0] == "dkey_01"


def test_page_to_table_filters_by_updated_after_date_object() -> None:
    rows, _, _ = _page_to_table(
        service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        updated_after=date(2026, 3, 24),
    )

    assert len(rows) == 1
    assert rows[0][0] == "dkey_01"


def test_find_detection_row_index() -> None:
    rows = [["dkey_00", "audio_00"], ["dkey_01", "audio_01"]]

    assert _find_detection_row_index(rows, "dkey_01") == 1
    assert _find_detection_row_index(rows, "missing") == 0


def test_selected_dataframe_row_index_prefers_selected_row_value() -> None:
    class FakeSelectEvent:
        index = (2, 0)
        row_value = ["dkey_02", "audio_02", "Species 2"]

    rows = [
        ["dkey_01", "audio_01", "Species 1"],
        ["dkey_02", "audio_02", "Species 2"],
    ]

    assert _selected_dataframe_row_index(rows, FakeSelectEvent()) == 1


def test_post_validation_queue_anchor_keeps_next_row_when_saved_row_leaves_filtered_page() -> None:
    rows = [["dkey_02", "audio_02"], ["dkey_03", "audio_03"]]

    assert _post_validation_queue_anchor(rows, "dkey_01", previous_index=0) == -1
    assert _post_validation_queue_anchor(rows, "dkey_02", previous_index=0) == 0


def test_advance_to_next_row_wraps_after_last_selected_row() -> None:
    service = FakeAudioService()
    rows = [
        ["dkey_01", "audio_01", "sp", 0.9, 0.0, 1.0],
        ["dkey_02", "audio_02", "sp", 0.8, 1.0, 2.0],
    ]

    selected_index, audio_path, cache_key, _, _, title = _advance_to_next_row_with_title(
        audio_service=service,
        dataset_repo="org/dataset",
        rows=rows,
        selected_index=1,
        cache_key="",
    )

    assert selected_index == 0
    assert audio_path == "/tmp/audio_01.wav"
    assert cache_key == "key:audio_01"
    assert title == "### Segment spectrogram"


def test_advance_to_next_row_wraps_to_first_pending_row() -> None:
    service = FakeAudioService()
    rows = [
        ["dkey_01", "audio_01", "sp", 0.9, 0.0, 1.0, "positive"],
        ["dkey_02", "audio_02", "sp", 0.8, 1.0, 2.0, "pending"],
        ["dkey_03", "audio_03", "sp", 0.7, 2.0, 3.0, "positive"],
    ]

    selected_index, audio_path, cache_key, _, _, _ = _advance_to_next_row_with_title(
        audio_service=service,
        dataset_repo="org/dataset",
        rows=rows,
        selected_index=2,
        cache_key="",
    )

    assert selected_index == 1
    assert audio_path == "/tmp/audio_02.wav"
    assert cache_key == "key:audio_02"


def test_autofetch_first_row_stops_when_filtered_species_is_complete() -> None:
    service = FakeAudioService()
    rows = [
        ["dkey_01", "audio_01", "sp", 0.9, 0.0, 1.0, "positive"],
        ["dkey_02", "audio_02", "sp", 0.8, 1.0, 2.0, "negative"],
    ]

    selected_index, audio_path, cache_key, status, spectrogram_path = _autofetch_first_row(
        audio_service=service,
        dataset_repo="org/dataset",
        rows=rows,
        cache_key="cache:old",
    )

    assert selected_index == -1
    assert audio_path is None
    assert cache_key == ""
    assert spectrogram_path is None
    assert "All segments for this species have been validated" in status


def test_selected_segment_card_shows_complete_species_message_only_when_all_rows_reviewed() -> None:
    complete_rows = [
        ["dkey_01", "audio_01", "sp", 0.9, 0.0, 1.0, "positive"],
        ["dkey_02", "audio_02", "sp", 0.8, 1.0, 2.0, "negative"],
    ]
    pending_rows = [
        ["dkey_01", "audio_01", "sp", 0.9, 0.0, 1.0, "pending"],
    ]

    complete_html = _selected_segment_card(complete_rows, -1)
    pending_html = _selected_segment_card(pending_rows, -1)

    assert "All segments for this species have been validated" in complete_html
    assert "Select a row manually to review or correct it." in complete_html
    assert "No segment loaded" in pending_html
    assert "All segments for this species have been validated" not in pending_html


def test_save_selected_validation_with_refresh_success() -> None:
    audio_service = FakeAudioService()
    validation_service = FakeValidationService()
    rows = [["dkey_01", "audio_11", "sp", 0.9, 0.0, 1.0, "pending", 0]]

    status, cache_key, audio_path, refreshed_rows, refreshed_page, refreshed_index, pending_status, conflict_key = _save_selected_validation_with_refresh(
        validation_service=validation_service,
        audio_service=audio_service,
        queue_service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        rows=rows,
        selected_index=0,
        status_value="positive",
        validator="validator-demo",
        notes="ok",
        cache_key="cache:audio_11",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        validator_filter="",
        status_filter="all",
        updated_after="",
        show_conflicts_only=False,
    )

    assert "Validation saved" in status
    assert cache_key == ""
    assert audio_path is None
    assert refreshed_page == 1
    assert refreshed_index == -1
    assert refreshed_rows[0][0] == "dkey_01"
    assert pending_status == ""
    assert conflict_key == ""


def test_reject_validation_requires_corrected_species_before_saving_or_advancing() -> None:
    audio_service = FakeAudioService()
    validation_service = FakeValidationService()
    rows = [["dkey_01", "audio_11", "sp", 0.9, 0.0, 1.0, "pending", 0]]

    status, cache_key, audio_output, refreshed_rows, refreshed_page, refreshed_index, pending_status, conflict_key = _save_selected_validation_with_refresh(
        validation_service=validation_service,
        audio_service=audio_service,
        queue_service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        rows=rows,
        selected_index=0,
        status_value="negative",
        validator="validator-demo",
        notes="wrong species",
        cache_key="cache:audio_11",
        page=1,
        scientific_name="sp",
        min_confidence=0.0,
        validator_filter="",
        status_filter="all",
        updated_after="",
        show_conflicts_only=False,
        corrected_species="",
    )
    advanced_index, advanced_audio, advanced_cache_key, advanced_status, spectrogram, title = _advance_after_validation_with_title(
        audio_service=audio_service,
        dataset_repo="org/dataset",
        rows=refreshed_rows,
        selected_index=refreshed_index,
        cache_key=cache_key,
        save_status=status,
    )

    assert status == VALIDATION_REJECT_CORRECTION_REQUIRED_STATUS
    assert validation_service.calls == []
    assert audio_service.cleaned == []
    assert cache_key == "cache:audio_11"
    assert refreshed_rows == rows
    assert refreshed_page == 1
    assert refreshed_index == 0
    assert pending_status == ""
    assert conflict_key == ""
    assert isinstance(audio_output, dict)
    assert advanced_index == 0
    assert isinstance(advanced_audio, dict)
    assert advanced_cache_key == "cache:audio_11"
    assert advanced_status == VALIDATION_REJECT_CORRECTION_REQUIRED_STATUS
    assert isinstance(spectrogram, dict)
    assert isinstance(title, dict)


def test_reject_validation_saves_corrected_species_when_provided() -> None:
    audio_service = FakeAudioService()
    validation_service = FakeValidationService()
    rows = [["dkey_01", "audio_11", "sp", 0.9, 0.0, 1.0, "pending", 0]]

    status, cache_key, _, _, _, _, _, _ = _save_selected_validation_with_refresh(
        validation_service=validation_service,
        audio_service=audio_service,
        queue_service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        rows=rows,
        selected_index=0,
        status_value="negative",
        validator="validator-demo",
        notes="wrong species",
        cache_key="cache:audio_11",
        page=1,
        scientific_name="sp",
        min_confidence=0.0,
        validator_filter="",
        status_filter="all",
        updated_after="",
        show_conflicts_only=False,
        corrected_species="Correct Species",
    )

    assert "Validation saved" in status
    assert cache_key == ""
    assert validation_service.calls[0]["status"] == "negative"
    assert validation_service.calls[0]["corrected_species"] == "Correct Species"


def test_corrected_species_ui_state_marks_error_and_resets_after_save() -> None:
    blocked_update, blocked_marker = _corrected_species_ui_after_validation(VALIDATION_REJECT_CORRECTION_REQUIRED_STATUS)
    saved_update, saved_marker = _corrected_species_ui_after_validation("Validation saved: dkey -> negative")
    conflict_update, conflict_marker = _corrected_species_ui_after_validation("Concurrency conflict: stale")

    assert isinstance(blocked_update, dict)
    assert 'data-error="true"' in blocked_marker
    assert saved_update["value"] is None
    assert 'data-error="false"' in saved_marker
    assert "value" not in conflict_update
    assert 'data-error="false"' in conflict_marker


def test_corrected_species_error_script_targets_marker_and_input() -> None:
    script = _corrected_species_required_script()
    payload = _corrected_species_error_payload(True)

    assert "bn-corrected-species-input" in script
    assert "bn-corrected-species-error-marker" in script
    assert "bn-corrected-species-error" in script
    assert 'data-error="true"' in payload


def test_save_selected_validation_advances_to_highest_confidence_pending_row() -> None:
    class QueueAfterSave:
        def list_all_detections(self, **kwargs: object) -> list[object]:
            _ = kwargs
            return [
                SimpleNamespace(
                    detection_key="dkey_selected",
                    audio_id="audio_selected",
                    scientific_name="sp",
                    confidence=0.99,
                    start_time=0,
                    end_time=1,
                ),
                SimpleNamespace(
                    detection_key="dkey_pending_high",
                    audio_id="audio_pending_high",
                    scientific_name="sp",
                    confidence=0.95,
                    start_time=1,
                    end_time=2,
                ),
                SimpleNamespace(
                    detection_key="dkey_pending_low",
                    audio_id="audio_pending_low",
                    scientific_name="sp",
                    confidence=0.60,
                    start_time=2,
                    end_time=3,
                ),
            ]

    class SnapshotAfterSave:
        def load_current_snapshot(self, project_slug: str, actor_username: str = "") -> dict[str, dict[str, object]]:
            _ = (project_slug, actor_username)
            return {"dkey_selected": {"status": "positive", "version": 1}}

    audio_service = FakeAudioService()
    status, cache_key, _, refreshed_rows, _, refreshed_index, _, _ = _save_selected_validation_with_refresh(
        validation_service=FakeValidationService(),
        audio_service=audio_service,
        queue_service=QueueAfterSave(),
        snapshot_reader=SnapshotAfterSave(),
        project_slug="demo-project",
        rows=[["dkey_selected", "audio_selected", "sp", 0.99, 0.0, 1.0, "pending", 0]],
        selected_index=0,
        status_value="positive",
        validator="validator-demo",
        notes="ok",
        cache_key="cache:audio_selected",
        page=1,
        scientific_name="sp",
        min_confidence=0.0,
        validator_filter="",
        status_filter="all",
        updated_after="",
        show_conflicts_only=False,
    )

    selected_index, audio_path, updated_cache_key, _, _, _ = _advance_to_next_row_with_title(
        audio_service=audio_service,
        dataset_repo="org/dataset",
        rows=refreshed_rows,
        selected_index=refreshed_index,
        cache_key=cache_key,
    )

    assert "Validation saved" in status
    assert [row[0] for row in refreshed_rows] == ["dkey_pending_high", "dkey_pending_low", "dkey_selected"]
    assert selected_index == 0
    assert audio_path == "/tmp/audio_pending_high.wav"
    assert updated_cache_key == "key:audio_pending_high"


def test_save_selected_validation_stops_when_species_is_complete() -> None:
    class QueueAfterFinalSave:
        def list_all_detections(self, **kwargs: object) -> list[object]:
            _ = kwargs
            return [
                SimpleNamespace(
                    detection_key="dkey_final",
                    audio_id="audio_final",
                    scientific_name="sp",
                    confidence=0.99,
                    start_time=0,
                    end_time=1,
                )
            ]

    class SnapshotAfterFinalSave:
        def load_current_snapshot(self, project_slug: str, actor_username: str = "") -> dict[str, dict[str, object]]:
            _ = (project_slug, actor_username)
            return {"dkey_final": {"status": "positive", "version": 1}}

    audio_service = FakeAudioService()
    status, cache_key, _, refreshed_rows, _, refreshed_index, _, _ = _save_selected_validation_with_refresh(
        validation_service=FakeValidationService(),
        audio_service=audio_service,
        queue_service=QueueAfterFinalSave(),
        snapshot_reader=SnapshotAfterFinalSave(),
        project_slug="demo-project",
        rows=[["dkey_final", "audio_final", "sp", 0.99, 0.0, 1.0, "pending", 0]],
        selected_index=0,
        status_value="positive",
        validator="validator-demo",
        notes="ok",
        cache_key="cache:audio_final",
        page=1,
        scientific_name="sp",
        min_confidence=0.0,
        validator_filter="",
        status_filter="all",
        updated_after="",
        show_conflicts_only=False,
    )

    selected_index, audio_path, updated_cache_key, audio_status, spectrogram_path, title = _advance_to_next_row_with_title(
        audio_service=audio_service,
        dataset_repo="org/dataset",
        rows=refreshed_rows,
        selected_index=refreshed_index,
        cache_key=cache_key,
    )

    assert "Validation saved" in status
    assert refreshed_index == -1
    assert selected_index == -1
    assert audio_path is None
    assert updated_cache_key == ""
    assert spectrogram_path is None
    assert title == "### Segment spectrogram"
    assert "All segments for this species have been validated" in audio_status


def test_save_selected_validation_with_refresh_conflict() -> None:
    audio_service = FakeAudioService()
    validation_service = FakeConflictValidationService()
    rows = [["dkey_01", "audio_11", "sp", 0.9, 0.0, 1.0, "pending", 0]]

    status, cache_key, audio_path, refreshed_rows, refreshed_page, refreshed_index, pending_status, conflict_key = _save_selected_validation_with_refresh(
        validation_service=validation_service,
        audio_service=audio_service,
        queue_service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        rows=rows,
        selected_index=0,
        status_value="positive",
        validator="validator-demo",
        notes="ok",
        cache_key="cache:audio_11",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        validator_filter="",
        status_filter="all",
        updated_after="",
        show_conflicts_only=False,
    )

    assert "Concurrency conflict" in status
    assert "Table reloaded" in status
    assert cache_key == "cache:audio_11"
    assert audio_path is None
    assert refreshed_page == 1
    assert refreshed_index == 0
    assert refreshed_rows[0][0] == "dkey_01"
    assert refreshed_rows[0][8] == "CONFLICT"
    assert refreshed_rows[0][9] == "HIGH"
    assert pending_status == "positive"
    assert conflict_key == "dkey_01"


def test_reapply_last_conflict_validation_with_refresh() -> None:
    audio_service = FakeAudioService()
    validation_service = FakeValidationService()
    rows = [["dkey_01", "audio_11", "sp", 0.9, 0.0, 1.0, "pending", 2, "conflict"]]

    status, cache_key, audio_path, refreshed_rows, refreshed_page, refreshed_index, pending_status, conflict_key = _reapply_last_conflict_validation_with_refresh(
        validation_service=validation_service,
        audio_service=audio_service,
        queue_service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        rows=rows,
        selected_index=0,
        pending_status_value="positive",
        conflict_detection_key="dkey_01",
        validator="validator-demo",
        notes="retry",
        cache_key="",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        validator_filter="",
        status_filter="all",
        updated_after="",
        show_conflicts_only=False,
    )

    assert "Validation saved" in status
    assert cache_key == ""
    assert audio_path is None
    assert refreshed_page == 1
    assert refreshed_index == -1
    assert refreshed_rows[0][0] == "dkey_01"
    assert pending_status == ""
    assert conflict_key == ""


def test_reapply_last_conflict_without_pending_status() -> None:
    audio_service = FakeAudioService()
    validation_service = FakeValidationService()
    rows = [["dkey_01", "audio_11", "sp", 0.9, 0.0, 1.0, "pending", 2, ""]]

    status, _, _, _, _, _, pending_status, conflict_key = _reapply_last_conflict_validation_with_refresh(
        validation_service=validation_service,
        audio_service=audio_service,
        queue_service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        rows=rows,
        selected_index=0,
        pending_status_value="",
        conflict_detection_key="",
        validator="validator-demo",
        notes="retry",
        cache_key="",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        validator_filter="",
        status_filter="all",
        updated_after="",
        show_conflicts_only=False,
    )

    assert "No pending validation" in status
    assert pending_status == ""
    assert conflict_key == ""


def test_create_app_with_keyboard_shortcuts() -> None:
    """Test that create_app successfully creates the UI with keyboard shortcuts enabled."""
    app = create_app()
    assert app is not None
    # Verify the app is a Gradio Blocks instance
    assert hasattr(app, "queue")
    assert hasattr(app, "launch")


def test_validation_shortcuts_are_validate_tab_scoped() -> None:
    script = _validation_shortcuts_script()

    assert "bn-validation-queue-table" in script
    assert "validateTabIsActive" in script
    assert "ArrowUp" in script
    assert "ArrowDown" in script
    assert "ArrowLeft" in script
    assert "ArrowRight" in script
    assert 'event.code === "Space"' in script
    assert "Backspace" not in script
    assert "bn-validate-confirm-btn" in script
    assert "bn-validate-reject-btn" in script
    assert "bn-validate-uncertain-btn" in script
    assert "bn-validate-skip-btn" in script
    assert "bn-validate-favorite-btn" in script
    assert "event.repeat" in script
    assert "isTypingTarget" in script


def test_batch_validate_conflicts_all_success() -> None:
    """Test batch approval of all conflicts in table."""
    audio_service = FakeAudioService()
    validation_service = FakeValidationService()
    rows = [
        ["dkey_01", "audio_11", "sp", 0.9, 0.0, 1.0, "pending", 1, "CONFLICT", "HIGH"],
        ["dkey_02", "audio_12", "sp", 0.85, 1.0, 2.0, "pending", 1, "CONFLICT", "HIGH"],
    ]

    status, cache_key, audio_path, refreshed_rows, refreshed_page = _batch_validate_conflicts(
        validation_service=validation_service,
        audio_service=audio_service,
        queue_service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        rows=rows,
        status_value="positive",
        validator="validator-demo",
        notes="batch approval",
        cache_key="",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        validator_filter="",
        status_filter="all",
        updated_after="",
    )

    assert "Processed 2 conflicts" in status
    assert "2 success" in status
    assert cache_key == ""
    assert refreshed_page == 1
    assert len(validation_service.calls) == 2


def test_batch_validate_conflicts_no_conflicts() -> None:
    """Test batch validation when no conflicts are present."""
    audio_service = FakeAudioService()
    validation_service = FakeValidationService()
    rows = [
        ["dkey_01", "audio_11", "sp", 0.9, 0.0, 1.0, "positive", 2, "", ""],
    ]

    status, cache_key, audio_path, refreshed_rows, refreshed_page = _batch_validate_conflicts(
        validation_service=validation_service,
        audio_service=audio_service,
        queue_service=FakeQueueService(),
        snapshot_reader=FakeSnapshotReader(),
        project_slug="demo-project",
        rows=rows,
        status_value="positive",
        validator="validator-demo",
        notes="batch approval",
        cache_key="",
        page=1,
        scientific_name="",
        min_confidence=0.0,
        validator_filter="",
        status_filter="all",
        updated_after="",
    )

    assert "No conflict detection" in status
    assert len(validation_service.calls) == 0


def test_load_seed_detections_from_json_dict(tmp_path: Path) -> None:
    payload = {
        "kenya-2024": [
            {
                "detection_key": "0000000000001001",
                "audio_id": "audio_1001",
                "scientific_name": "Cyanocorax cyanopogon",
                "confidence": 0.91,
                "start_time": 0.0,
                "end_time": 1.0,
            }
        ]
    }
    seed_file = tmp_path / "detections.json"
    seed_file.write_text(json.dumps(payload), encoding="utf-8")

    result = _load_seed_detections(str(seed_file))

    assert "kenya-2024" in result
    assert len(result["kenya-2024"]) == 1
    assert result["kenya-2024"][0].audio_id == "audio_1001"


def test_validate_seed_file_warns_for_invalid_shape(tmp_path: Path) -> None:
    seed_file = tmp_path / "detections-invalid.json"
    seed_file.write_text(json.dumps({"kenya-2024": {"wrong": True}}), encoding="utf-8")

    warning = _validate_seed_file(str(seed_file))

    assert "Invalid" in warning


def test_validate_seed_file_warns_when_file_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-seed.json"

    warning = _validate_seed_file(str(missing_path))

    assert "not found" in warning
    assert "BIRDNET_DETECTIONS_FILE" in warning


def test_build_detection_repository_includes_new_project_defaults() -> None:
    queue, warning = _build_detection_repository(["brand-new-project"], seed_file_path=None)
    page = queue.get_page(project_slug="brand-new-project", page=1, page_size=10)

    assert warning == ""
    assert page.project_slug == "brand-new-project"
    assert len(page.items) == 4


def test_get_project_detection_count_reads_total_items() -> None:
    queue = FakeQueueService()

    total = _get_project_detection_count(queue, "kenya-2024")

    assert total == 2


def test_get_project_detection_count_handles_service_error() -> None:
    class BrokenQueueService:
        def get_page(self, **kwargs: object):
            _ = kwargs
            raise RuntimeError("boom")

    total = _get_project_detection_count(BrokenQueueService(), "kenya-2024")

    assert total == 0


def test_build_queue_badge_without_project() -> None:
    badge = _build_queue_badge(FakeQueueService(), None)

    assert "Queue: --" in badge


def test_build_queue_badge_with_project() -> None:
    badge = _build_queue_badge(FakeQueueService(), "kenya-2024")

    assert "Queue: 2" in badge


def test_fetch_selected_audio_repo_hint() -> None:
    service = FakeAudioService()
    rows = [["k1", "audio_03", "sp", 0.9, 0.0, 1.0]]

    _, _, status = _fetch_selected_audio(
        audio_service=service,
        dataset_repo="",
        rows=rows,
        selected_index=0,
        previous_cache_key="",
    )

    assert "owner/repo" in status
    assert "Example" in status


def test_load_projects_from_file_reads_valid_payload(tmp_path: Path) -> None:
    payload = [
        {
            "project_slug": "project-a",
            "name": "Project A",
            "dataset_repo_id": "org/project-a",
            "state_backend": "hf_project_store",
            "state_repo_id": "org/project-a_state",
            "state_schema_version": 1,
            "state_status": "ready",
            "active": True,
        }
    ]
    path = tmp_path / "projects.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    projects = _load_projects_from_file(str(path))

    assert len(projects) == 1
    assert projects[0].project_slug == "project-a"
    assert projects[0].state_backend == "hf_project_store"
    assert projects[0].state_repo_id == "org/project-a_state"
    assert projects[0].state_status == "ready"


def test_load_user_access_from_file_reads_valid_payload(tmp_path: Path) -> None:
    payload = {
        "validator_a": {"project-a": "validator"},
        "admin_a": {"project-a": "admin", "project-b": "admin"},
    }
    path = tmp_path / "access.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    access = _load_user_access_from_file(str(path))

    assert access["validator_a"]["project-a"].value == "validator"
    assert access["admin_a"]["project-b"].value == "admin"


def test_persist_bootstrap_state_blocks_unplanned_project_removal(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.json"
    users_file = tmp_path / "user_access.json"
    invites_file = tmp_path / "invites.json"
    projects_file.write_text(
        json.dumps(
            [
                {
                    "project_id": "project-a-id",
                    "project_slug": "project-a",
                    "name": "Project A",
                    "dataset_repo_id": "org/project-a",
                    "active": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    users_file.write_text(json.dumps({"owner": {"project-a": "admin"}}), encoding="utf-8")

    auth_service = AuthService()
    auth_service.upsert_user_project_role("owner", "project-b", Role.admin)
    admin_manager = AdminPanelManager(auth_service, invite_notifier=NoopInviteNotifier())
    admin_manager.register_project(
        Project(
            project_slug="project-b",
            name="Project B",
            dataset_repo_id="org/project-b",
            owner_username="owner",
        )
    )

    with pytest.raises(StateSafetyError):
        _persist_bootstrap_state(
            projects_path=projects_file,
            user_access_path=users_file,
            invites_path=invites_file,
            admin_manager=admin_manager,
            auth_service=auth_service,
        )

    assert json.loads(projects_file.read_text(encoding="utf-8"))[0]["project_slug"] == "project-a"


def test_persist_bootstrap_state_allows_explicit_delete_and_creates_backups(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.json"
    users_file = tmp_path / "user_access.json"
    invites_file = tmp_path / "invites.json"
    projects_file.write_text(
        json.dumps(
            [
                {
                    "project_id": "project-a-id",
                    "project_slug": "project-a",
                    "name": "Project A",
                    "dataset_repo_id": "org/project-a",
                    "active": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    users_file.write_text(json.dumps({"owner": {"project-a": "admin"}}), encoding="utf-8")
    invites_file.write_text(json.dumps({}), encoding="utf-8")

    auth_service = AuthService()
    admin_manager = AdminPanelManager(auth_service, invite_notifier=NoopInviteNotifier())

    _persist_bootstrap_state(
        projects_path=projects_file,
        user_access_path=users_file,
        invites_path=invites_file,
        admin_manager=admin_manager,
        auth_service=auth_service,
        allowed_removed_project_slugs={"project-a"},
    )

    assert json.loads(projects_file.read_text(encoding="utf-8")) == []
    assert json.loads(users_file.read_text(encoding="utf-8")) == {}
    assert list((tmp_path / ".backups").glob("projects.json.*.bak"))
    assert list((tmp_path / ".backups").glob("user_access.json.*.bak"))


def test_persist_bootstrap_state_writes_project_state_metadata(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.json"
    users_file = tmp_path / "user_access.json"
    invites_file = tmp_path / "invites.json"
    projects_file.write_text(json.dumps([]), encoding="utf-8")
    users_file.write_text(json.dumps({}), encoding="utf-8")
    invites_file.write_text(json.dumps({}), encoding="utf-8")

    auth_service = AuthService()
    auth_service.upsert_user_project_role("owner", "project-a", Role.admin)
    admin_manager = AdminPanelManager(auth_service, invite_notifier=NoopInviteNotifier())
    admin_manager.register_project(
        Project(
            project_slug="project-a",
            name="Project A",
            dataset_repo_id="org/project-a",
            owner_username="owner",
            state_backend="hf_project_store",
            state_repo_id="org/project-a_state",
            state_status="ready",
        )
    )

    _persist_bootstrap_state(
        projects_path=projects_file,
        user_access_path=users_file,
        invites_path=invites_file,
        admin_manager=admin_manager,
        auth_service=auth_service,
    )

    payload = json.loads(projects_file.read_text(encoding="utf-8"))
    assert payload[0]["state_backend"] == "hf_project_store"
    assert payload[0]["state_repo_id"] == "org/project-a_state"
    assert payload[0]["state_schema_version"] == 1
    assert payload[0]["state_status"] == "ready"


def test_bootstrap_auth_and_projects_uses_config_files_without_demo_fallback(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.json"
    projects_file.write_text(
        json.dumps(
            [
                {
                    "project_slug": "project-a",
                    "name": "Project A",
                    "dataset_repo_id": "org/project-a",
                    "active": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    users_file = tmp_path / "users.json"
    users_file.write_text(
        json.dumps({"validator_a": {"project-a": "validator"}}),
        encoding="utf-8",
    )

    runtime_config = RuntimeConfig(
        detection_seed_path=None,
        validation_base_dir=str(tmp_path / "validations"),
        bootstrap_base_dir=str(tmp_path / "bootstrap"),
        page_size=25,
        projects_file_path=str(projects_file),
        user_access_file_path=str(users_file),
        invites_file_path=None,
        invite_ttl_hours=72,
        enable_demo_bootstrap=False,
        invite_email_enabled=False,
        invite_email_sender="",
        invite_email_login_url="",
    )
    auth_service = AuthService()
    from src.services.invite_email_notifier import EmailJSInviteEmailNotifier
    notifier = EmailJSInviteEmailNotifier("", "", "", "", timeout_seconds=20)
    admin_manager = AdminPanelManager(auth_service, invite_notifier=notifier)

    warning = _bootstrap_auth_and_projects(auth_service, admin_manager, runtime_config)
    emergency_admin_session = auth_service.login("admin_user")

    assert "Emergency admin access" in warning
    assert auth_service.login("validator_a") is not None
    assert emergency_admin_session is not None
    assert any(p["project_slug"] == "project-a" for p in admin_manager.list_projects())


def test_bootstrap_auth_and_projects_warns_when_not_configured(tmp_path: Path) -> None:
    runtime_config = RuntimeConfig(
        detection_seed_path=None,
        validation_base_dir=str(tmp_path / "validations"),
        bootstrap_base_dir=str(tmp_path / "bootstrap"),
        page_size=25,
        projects_file_path=None,
        user_access_file_path=None,
        invites_file_path=None,
        invite_ttl_hours=72,
        enable_demo_bootstrap=False,
        invite_email_enabled=False,
        invite_email_sender="",
        invite_email_login_url="",
    )
    auth_service = AuthService()
    from src.services.invite_email_notifier import EmailJSInviteEmailNotifier
    notifier = EmailJSInviteEmailNotifier("", "", "", "", timeout_seconds=20)
    admin_manager = AdminPanelManager(auth_service, invite_notifier=notifier)

    warning = _bootstrap_auth_and_projects(auth_service, admin_manager, runtime_config)
    assert warning == ""


def test_bootstrap_auth_and_projects_uses_demo_bootstrap_when_enabled(tmp_path: Path) -> None:
    runtime_config = RuntimeConfig(
        detection_seed_path=None,
        validation_base_dir=str(tmp_path / "validations"),
        bootstrap_base_dir=str(tmp_path / "bootstrap"),
        page_size=25,
        projects_file_path=None,
        user_access_file_path=None,
        invites_file_path=None,
        invite_ttl_hours=72,
        enable_demo_bootstrap=True,
        invite_email_enabled=False,
        invite_email_sender="",
        invite_email_login_url="",
    )
    auth_service = AuthService()
    from src.services.invite_email_notifier import EmailJSInviteEmailNotifier
    notifier = EmailJSInviteEmailNotifier("", "", "", "", timeout_seconds=20)
    admin_manager = AdminPanelManager(auth_service, invite_notifier=notifier)

    warning = _bootstrap_auth_and_projects(auth_service, admin_manager, runtime_config)

    assert warning == ""
    assert auth_service.login("demo_user") is not None
    assert auth_service.login("admin_user") is not None
    assert any(p["project_slug"] == "demo-project" for p in admin_manager.list_projects())


def test_bootstrap_auth_and_projects_loads_configured_hf_project_state(tmp_path: Path) -> None:
    class FakeStateLoader:
        def load_project_state(self, *, state_repo_id: str, token: str):  # noqa: ANN001
            assert state_repo_id == "owner/project-a_state"
            assert token == "hf_read"
            project = Project(
                project_slug="project-a",
                name="Project A",
                dataset_repo_id="owner/project-a",
                owner_username="owner",
                state_backend="hf_project_store",
                state_repo_id=state_repo_id,
            )
            return HfProjectStateStoreLoadedProject(
                state_repo_id=state_repo_id,
                project=project,
                user_access={"owner": {"project-a": Role.admin}, "validator": {"project-a": Role.validator}},
                pending_invites={
                    "pending-user": {
                        "project-a": {
                            "role": "validator",
                            "invited_by": "owner",
                            "created_at": "2026-05-24T12:00:00+00:00",
                            "expires_at": "2099-05-24T12:00:00+00:00",
                            "username": "pending-user",
                            "invitee_email": "",
                        }
                    }
                },
            )

    runtime_config = RuntimeConfig(
        detection_seed_path=None,
        validation_base_dir=str(tmp_path / "validations"),
        bootstrap_base_dir=str(tmp_path / "bootstrap"),
        page_size=25,
        projects_file_path=None,
        user_access_file_path=None,
        invites_file_path=None,
        invite_ttl_hours=72,
        enable_demo_bootstrap=False,
        invite_email_enabled=False,
        invite_email_sender="",
        invite_email_login_url="",
        hf_project_state_repos=("owner/project-a_state",),
    )
    auth_service = AuthService()
    admin_manager = AdminPanelManager(auth_service, invite_notifier=NoopInviteNotifier())

    warning = _bootstrap_auth_and_projects(
        auth_service,
        admin_manager,
        runtime_config,
        hf_project_state_token="hf_read",
        hf_project_state_loader=FakeStateLoader(),
    )

    assert warning == ""
    assert admin_manager.get_project("project-a") is not None
    assert auth_service.login("owner") is not None
    assert auth_service.get_user_role_for_project("validator", "project-a") == Role.validator
    assert len(auth_service.list_pending_invites("pending-user")) == 1


def test_bootstrap_auth_and_projects_reports_hf_project_state_load_errors(tmp_path: Path) -> None:
    runtime_config = RuntimeConfig(
        detection_seed_path=None,
        validation_base_dir=str(tmp_path / "validations"),
        bootstrap_base_dir=str(tmp_path / "bootstrap"),
        page_size=25,
        projects_file_path=None,
        user_access_file_path=None,
        invites_file_path=None,
        invite_ttl_hours=72,
        enable_demo_bootstrap=False,
        invite_email_enabled=False,
        invite_email_sender="",
        invite_email_login_url="",
        hf_project_state_repos=("owner/project-a_state",),
    )
    auth_service = AuthService()
    admin_manager = AdminPanelManager(auth_service, invite_notifier=NoopInviteNotifier())

    warning = _bootstrap_auth_and_projects(auth_service, admin_manager, runtime_config)

    assert "no HF token" in warning


def test_bootstrap_admin_storage_ignores_discovered_state_repo_without_manifest(tmp_path: Path) -> None:
    class MissingManifestLoader:
        def load_project_state(self, *, state_repo_id: str, token: str):  # noqa: ANN001
            _ = (state_repo_id, token)
            raise HfProjectStateStoreError(
                "Could not read project.json from jrrribeiro/upload_test1_state: "
                "404 Client Error. Entry Not Found"
            )

    runtime_config = RuntimeConfig(
        detection_seed_path=None,
        validation_base_dir=str(tmp_path / "validations"),
        bootstrap_base_dir=str(tmp_path / "bootstrap"),
        page_size=25,
        projects_file_path=None,
        user_access_file_path=None,
        invites_file_path=None,
        invite_ttl_hours=72,
        enable_demo_bootstrap=False,
        invite_email_enabled=False,
        invite_email_sender="",
        invite_email_login_url="",
        hf_admin_storage_mode_enabled=True,
        hf_project_state_repos=("jrrribeiro/upload_test1_state",),
    )
    auth_service = AuthService()
    admin_manager = AdminPanelManager(auth_service, invite_notifier=NoopInviteNotifier())

    warning = _bootstrap_auth_and_projects(
        auth_service,
        admin_manager,
        runtime_config,
        hf_project_state_token="hf_storage",
        hf_project_state_loader=MissingManifestLoader(),  # type: ignore[arg-type]
    )

    assert "upload_test1_state" not in warning
    assert "project.json" not in warning


def test_bootstrap_auth_and_projects_recovers_emergency_admin_when_missing(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.json"
    projects_file.write_text(
        json.dumps(
            [
                {
                    "project_slug": "project-a",
                    "name": "Project A",
                    "dataset_repo_id": "org/project-a",
                    "active": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    users_file = tmp_path / "users.json"
    users_file.write_text(
        json.dumps({"validator_only": {"project-a": "validator"}}),
        encoding="utf-8",
    )

    runtime_config = RuntimeConfig(
        detection_seed_path=None,
        validation_base_dir=str(tmp_path / "validations"),
        bootstrap_base_dir=str(tmp_path / "bootstrap"),
        page_size=25,
        projects_file_path=str(projects_file),
        user_access_file_path=str(users_file),
        invites_file_path=None,
        invite_ttl_hours=72,
        enable_demo_bootstrap=False,
        invite_email_enabled=False,
        invite_email_sender="",
        invite_email_login_url="",
    )
    auth_service = AuthService()
    from src.services.invite_email_notifier import EmailJSInviteEmailNotifier
    notifier = EmailJSInviteEmailNotifier("", "", "", "", timeout_seconds=20)
    admin_manager = AdminPanelManager(auth_service, invite_notifier=notifier)

    warning = _bootstrap_auth_and_projects(auth_service, admin_manager, runtime_config)
    emergency_session = auth_service.login("admin_user")

    assert "Emergency admin access" in warning
    assert emergency_session is not None
    assert emergency_session.role.value == "admin"


def test_auto_auth_policy_requires_oauth_in_space_without_demo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SPACE_ID", "owner/space")
    runtime_config = RuntimeConfig(
        detection_seed_path=None,
        validation_base_dir=str(tmp_path / "validations"),
        bootstrap_base_dir=str(tmp_path / "bootstrap"),
        page_size=25,
        projects_file_path=None,
        user_access_file_path=None,
        invites_file_path=None,
        invite_ttl_hours=72,
        enable_demo_bootstrap=False,
        invite_email_enabled=False,
        invite_email_sender="",
        invite_email_login_url="",
    )

    allow_username, label, description = _resolve_username_login_policy(runtime_config)

    assert allow_username is False
    assert "OAuth" in label
    assert "OAuth" in description


def test_auto_auth_policy_allows_username_for_demo(tmp_path: Path) -> None:
    runtime_config = RuntimeConfig(
        detection_seed_path=None,
        validation_base_dir=str(tmp_path / "validations"),
        bootstrap_base_dir=str(tmp_path / "bootstrap"),
        page_size=25,
        projects_file_path=None,
        user_access_file_path=None,
        invites_file_path=None,
        invite_ttl_hours=72,
        enable_demo_bootstrap=True,
        invite_email_enabled=False,
        invite_email_sender="",
        invite_email_login_url="",
    )

    allow_username, label, _ = _resolve_username_login_policy(runtime_config)

    assert allow_username is True
    assert "username" in label


def test_load_dataset_detections_for_project_reads_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from src.ui import app_factory as module

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            _ = token

        def list_repo_files(self, repo_id: str, repo_type: str = "dataset") -> list[str]:
            _ = repo_id
            _ = repo_type
            return ["detections.jsonl"]

    metadata_file = tmp_path / "detections.jsonl"
    metadata_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "project_slug": "project-a",
                        "audio_id": "audio_0001",
                        "scientific_name": "Species A",
                        "confidence": 0.92,
                        "start_time": 1.0,
                        "end_time": 2.0,
                    }
                ),
                json.dumps(
                    {
                        "project_slug": "project-a",
                        "audio_id": "audio_0002",
                        "scientific_name": "Species B",
                        "confidence": 0.81,
                        "start_time": 2.0,
                        "end_time": 3.0,
                    }
                ),
                json.dumps(
                    {
                        "project_slug": "project-other",
                        "audio_id": "audio_9999",
                        "scientific_name": "Other",
                        "confidence": 0.5,
                        "start_time": 0.0,
                        "end_time": 1.0,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "HfApi", FakeHfApi)
    monkeypatch.setattr(module, "hf_hub_download", lambda **kwargs: str(metadata_file))

    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="org/project-a",
        active=True,
    )
    detections, warning = module._load_dataset_detections_for_project(project)

    assert warning == ""
    assert len(detections) == 2
    assert {item.scientific_name for item in detections} == {"Species A", "Species B"}


def test_load_dataset_detections_for_project_uses_fallback_hf_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.ui import app_factory as module

    observed_tokens: list[str | None] = []

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            observed_tokens.append(token)

        def list_repo_files(self, repo_id: str, repo_type: str = "dataset") -> list[str]:
            _ = repo_id
            _ = repo_type
            return ["detections.jsonl"]

    metadata_file = tmp_path / "detections.jsonl"
    metadata_file.write_text(
        json.dumps(
            {
                "audio_id": "audio_0001",
                "scientific_name": "Species A",
                "confidence": 0.92,
                "start_time": 1.0,
                "end_time": 2.0,
            }
        ),
        encoding="utf-8",
    )

    def fake_download(**kwargs):
        observed_tokens.append(kwargs.get("token"))
        if kwargs.get("filename") != "detections.jsonl":
            raise FileNotFoundError(str(kwargs.get("filename")))
        return str(metadata_file)

    monkeypatch.setattr(module, "HfApi", FakeHfApi)
    monkeypatch.setattr(module, "hf_hub_download", fake_download)

    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="org/project-a",
        active=True,
    )
    detections, warning = module._load_dataset_detections_for_project(project, hf_token="hf_session")

    assert warning == ""
    assert len(detections) == 1
    assert observed_tokens[-2:] == ["hf_session", "hf_session"]


def test_load_dataset_detections_for_project_reports_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui import app_factory as module

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            _ = token

        def list_repo_files(self, repo_id: str, repo_type: str = "dataset") -> list[str]:
            _ = repo_id
            _ = repo_type
            raise RuntimeError("401 Client Error")

    monkeypatch.setattr(module, "HfApi", FakeHfApi)

    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="org/private-project",
        active=True,
    )
    detections, warning = module._load_dataset_detections_for_project(project)

    assert detections == []
    assert "No Hugging Face token is configured" in warning
    assert "Admin > Project token management" in warning


def test_resolve_project_fetch_token_uses_session_then_project_then_env(monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="org/project-a",
        dataset_token="hf_project",
        active=True,
    )

    assert _resolve_project_fetch_token(project, "hf_session") == "hf_session"
    assert _resolve_project_fetch_token(project, None) == "hf_project"

    project.dataset_token = None
    monkeypatch.setenv("HF_TOKEN", "hf_env")

    assert _resolve_project_fetch_token(project, None) == "hf_env"


def test_build_detection_repository_prefers_dataset_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui import app_factory as module

    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="org/project-a",
        active=True,
    )

    monkeypatch.setattr(
        module,
        "_load_dataset_detections_for_project",
        lambda project_obj, hf_token=None: (
            [
                Detection(
                    detection_key="0000000000002222",
                    audio_id="audio_dataset_1",
                    scientific_name="Dataset Species",
                    confidence=0.95,
                    start_time=0.0,
                    end_time=1.0,
                )
            ],
            "",
        ),
    )

    queue, warning = _build_detection_repository(
        ["project-a"],
        seed_file_path=None,
        project_map={"project-a": project},
        allow_demo_defaults=False,
    )
    page = queue.get_page(project_slug="project-a", page=1, page_size=10)

    assert warning == ""
    assert len(page.items) == 1
    assert page.items[0].scientific_name == "Dataset Species"


def test_load_dataset_detections_for_project_falls_back_to_audiofolder_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui import app_factory as module

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            _ = token

        def list_repo_files(self, repo_id: str, repo_type: str = "dataset") -> list[str]:
            _ = repo_id
            _ = repo_type
            return [
                "audio/segments/Accipiter_striatus/Catim_20250221_060600_0.0-3.0s_85%.wav",
                "audio/segments/Aegolius_harrisii/Aiuab_20260123_182900_9.0-12.0s_68%.wav",
            ]

    monkeypatch.setattr(module, "HfApi", FakeHfApi)

    project = Project(
        project_slug="teste7",
        name="Teste 7",
        dataset_repo_id="jrrribeiro/teste7",
        active=True,
    )

    detections, warning = module._load_dataset_detections_for_project(project)

    assert warning == ""
    assert len(detections) == 2
    assert {item.scientific_name for item in detections} == {"Accipiter striatus", "Aegolius harrisii"}
    assert {item.audio_id for item in detections} == {
        "segments/Accipiter_striatus/Catim_20250221_060600_0.0-3.0s_85%.wav",
        "segments/Aegolius_harrisii/Aiuab_20260123_182900_9.0-12.0s_68%.wav",
    }


def test_parse_segment_filename_hint_reads_time_and_confidence() -> None:
    from src.ui import app_factory as module

    start, end, conf = module._parse_segment_filename_hint("any_12.0-15.0s_85%.wav")

    assert start == 12.0
    assert end == 15.0
    assert conf == 0.85


def test_parse_segment_filename_hint_reads_uploader_hash_filename() -> None:
    from src.ui import app_factory as module

    start, end, conf = module._parse_segment_filename_hint(
        "Catim_LS-1500_U10_20250224_063500_33.0-36.0s_95__cf04dd382267.wav"
    )

    assert start == 33.0
    assert end == 36.0
    assert conf == 0.95


def test_build_detection_from_row_prefers_segment_path_for_audio_id() -> None:
    from src.ui import app_factory as module

    row = {
        "project_slug": "project-a",
        "segment_path_in_repo": "audio/segments/species_a/example.wav",
        "audio_id": "source_stem_only",
        "scientific_name": "Species A",
        "confidence": 0.77,
        "start_time": 0.0,
        "end_time": 3.0,
    }

    detection = module._build_detection_from_row(row, 0, "project-a")

    assert detection is not None
    assert detection.audio_id == "segments/species_a/example.wav"


def test_build_detection_from_row_accepts_birdnet_column_variants() -> None:
    from src.ui import app_factory as module

    row = {
        "segment_path_in_repo": "audio/segments/Accipiter_striatus/example_33.0-36.0s_95__abc123def456.wav",
        "Scientific Name": "Accipiter striatus",
        "Confidence": 0.731,
        "Start_tim": 33.0,
        "End_time": 36.0,
    }

    detection = module._build_detection_from_row(row, 0, "project-a")

    assert detection is not None
    assert detection.scientific_name == "Accipiter striatus"
    assert detection.confidence == 0.731
    assert detection.start_time == 33.0
    assert detection.end_time == 36.0


def test_load_dataset_detections_for_project_uses_parquet_shards_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui import app_factory as module

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            _ = token

        def list_repo_files(self, repo_id: str, repo_type: str = "dataset") -> list[str]:
            _ = repo_id
            _ = repo_type
            return ["manifest.json", "index/shards/shard-00000.parquet"]

    expected = [
        Detection(
            detection_key="0000000000003333",
            audio_id="segments/species_a/example.wav",
            scientific_name="Species A",
            confidence=0.9,
            start_time=0.0,
            end_time=3.0,
        )
    ]

    monkeypatch.setattr(module, "HfApi", FakeHfApi)
    monkeypatch.setattr(
        module,
        "_load_detections_from_parquet_shards",
        lambda project, dataset_repo, token, repo_files: (expected, ""),
    )

    project = Project(
        project_slug="project-a",
        name="Project A",
        dataset_repo_id="org/project-a",
        active=True,
    )

    detections, warning = module._load_dataset_detections_for_project(project)

    assert warning == ""
    assert len(detections) == 1
    assert detections[0].scientific_name == "Species A"


def test_load_dataset_detections_for_project_uses_files_parquet_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.ui import app_factory as module

    files_index = tmp_path / "files.parquet"
    pd.DataFrame.from_records(
        [
            {
                "stored_path": "audio/Accipiter_striatus/shard-000000/Catim_20250221_060600_0.0-3.0s_85__abc123.wav",
                "original_relative_path": "Accipiter striatus/Catim_20250221_060600_0.0-3.0s_85%.wav",
                "logical_group": "Accipiter striatus",
                "filename": "Catim_20250221_060600_0.0-3.0s_85%.wav",
                "size": 123,
            }
        ]
    ).to_parquet(files_index, index=False)

    detections_csv = tmp_path / "detections.csv"
    detections_csv.write_text(
        "source_file,scientific_name,confidence,start_time,end_time\n"
        "Catim_20250221_060600_0.0-3.0s_85%.wav,Accipiter striatus,0.85,0,3\n",
        encoding="utf-8",
    )

    class FakeHfApi:
        def __init__(self, token: str | None = None) -> None:
            _ = token

        def list_repo_files(self, repo_id: str, repo_type: str = "dataset") -> list[str]:
            _ = repo_id
            _ = repo_type
            return ["index/files.parquet", "index/detections.csv"]

    def fake_download(repo_id: str, repo_type: str, filename: str, token: str | None = None) -> str:
        _ = repo_id
        _ = repo_type
        _ = token
        if filename == "index/files.parquet":
            return str(files_index)
        if filename == "index/detections.csv":
            return str(detections_csv)
        raise FileNotFoundError(filename)

    monkeypatch.setattr(module, "HfApi", FakeHfApi)
    monkeypatch.setattr(module, "hf_hub_download", fake_download)

    project = Project(
        project_slug="ppbio-rabeca",
        name="PPBIO RABECA",
        dataset_repo_id="jrrribeiro/PPBIO-RABECA",
        active=True,
    )

    detections, warning = module._load_dataset_detections_for_project(project)

    assert warning == ""
    assert len(detections) == 1
    assert detections[0].audio_id == "Accipiter_striatus/shard-000000/Catim_20250221_060600_0.0-3.0s_85__abc123.wav"
    assert detections[0].scientific_name == "Accipiter striatus"
    assert detections[0].confidence == 0.85
    assert detections[0].start_time == 0.0
    assert detections[0].end_time == 3.0


def test_files_index_merges_detection_csv_by_source_species_and_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.ui import app_factory as module

    original_relative_path = "Accipiter striatus/Catim_LS-1500_U10_20250224_063500_33.0-36.0s_95%.wav"
    files_index = tmp_path / "files.parquet"
    pd.DataFrame.from_records(
        [
            {
                "stored_path": "audio/Accipiter_striatus/shard-000001/Catim_LS-1500_U10_20250224_063500_33.0-36.0s_95__cf04dd382267.wav",
                "original_relative_path": original_relative_path,
                "logical_group": "Accipiter striatus",
                "filename": "Catim_LS-1500_U10_20250224_063500_33.0-36.0s_95%.wav",
                "size": 123,
            }
        ]
    ).to_parquet(files_index, index=False)

    detections_csv = tmp_path / "detections.csv"
    detections_csv.write_text(
        "source_file,scientific_name,confidence,start_time,end_time\n"
        "Catim_LS-1500_U10_20250224_063500.wav,Accipiter striatus,0.812345,33,36\n",
        encoding="utf-8",
    )

    class TreeApiShouldNotRun:
        def __init__(self, token: str | None = None) -> None:
            _ = token

        def list_repo_files(self, repo_id: str, repo_type: str = "dataset") -> list[str]:
            _ = repo_id
            _ = repo_type
            raise AssertionError("tree discovery should not run when index/files.parquet is available")

    def fake_download(repo_id: str, repo_type: str, filename: str, token: str | None = None) -> str:
        _ = repo_id
        _ = repo_type
        _ = token
        if filename == "index/files.parquet":
            return str(files_index)
        if filename == "index/detections.csv":
            return str(detections_csv)
        raise FileNotFoundError(filename)

    monkeypatch.setattr(module, "HfApi", TreeApiShouldNotRun)
    monkeypatch.setattr(module, "hf_hub_download", fake_download)

    project = Project(
        project_slug="ppbio-rabeca",
        name="PPBIO RABECA",
        dataset_repo_id="jrrribeiro/PPBIO-RABECA",
        active=True,
    )

    detections, warning = module._load_dataset_detections_for_project(project)

    assert warning == ""
    assert len(detections) == 1
    assert detections[0].detection_key == hashlib.sha1(original_relative_path.encode("utf-8")).hexdigest()[:16]
    assert detections[0].audio_id == "Accipiter_striatus/shard-000001/Catim_LS-1500_U10_20250224_063500_33.0-36.0s_95__cf04dd382267.wav"
    assert detections[0].confidence == 0.812345
    assert detections[0].start_time == 33.0
    assert detections[0].end_time == 36.0
    assert detections[0].source_metadata["segment_uploader_key"] == "cf04dd382267"


def test_load_dataset_detections_for_project_reads_known_files_index_before_repo_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.ui import app_factory as module

    files_index = tmp_path / "files.parquet"
    pd.DataFrame.from_records(
        [
            {
                "stored_path": "audio/Accipiter_striatus/shard-000000/example.wav",
                "original_relative_path": "Accipiter striatus/example.wav",
                "logical_group": "Accipiter striatus",
                "filename": "example.wav",
                "size": 123,
            }
        ]
    ).to_parquet(files_index, index=False)

    class TreeApiShouldNotRun:
        def __init__(self, token: str | None = None) -> None:
            _ = token

        def list_repo_files(self, repo_id: str, repo_type: str = "dataset") -> list[str]:
            _ = repo_id
            _ = repo_type
            raise AssertionError("tree discovery should not run when index/files.parquet is available")

    def fake_download(repo_id: str, repo_type: str, filename: str, token: str | None = None) -> str:
        _ = repo_id
        _ = repo_type
        _ = token
        if filename == "index/files.parquet":
            return str(files_index)
        raise FileNotFoundError(filename)

    monkeypatch.setattr(module, "HfApi", TreeApiShouldNotRun)
    monkeypatch.setattr(module, "hf_hub_download", fake_download)

    project = Project(
        project_slug="ppbio-rabeca",
        name="PPBIO RABECA",
        dataset_repo_id="jrrribeiro/PPBIO-RABECA",
        active=True,
    )

    detections, warning = module._load_dataset_detections_for_project(project)

    assert warning == ""
    assert len(detections) == 1
    assert detections[0].audio_id == "Accipiter_striatus/shard-000000/example.wav"


def test_load_dataset_detections_for_project_explains_tree_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.ui import app_factory as module

    monkeypatch.setattr(module, "_load_detections_from_known_files_index", lambda **kwargs: ([], ""))

    class RateLimitedHfApi:
        def __init__(self, token: str | None = None) -> None:
            _ = token

        def list_repo_files(self, repo_id: str, repo_type: str = "dataset") -> list[str]:
            _ = repo_id
            _ = repo_type
            raise RuntimeError("429 Client Error: Too Many Requests")

    monkeypatch.setattr(module, "HfApi", RateLimitedHfApi)

    project = Project(
        project_slug="ppbio-rabeca",
        name="PPBIO RABECA",
        dataset_repo_id="jrrribeiro/PPBIO-RABECA",
        active=True,
    )

    detections, warning = module._load_dataset_detections_for_project(project)

    assert detections == []
    assert "rate-limited dataset discovery" in warning
    assert "fast HF_Dataset_Uploader index path" in warning
