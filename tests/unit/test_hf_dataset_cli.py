import json
from pathlib import Path

import pandas as pd
import pytest

from cli.hf_dataset_cli import (
    SCHEMA_VERSION,
    _chunk_items,
    _load_resume_state,
    _save_resume_state,
    build_manifest_payload,
    build_shards_in_directory,
    collect_verify_errors,
    discover_audio_files,
    load_detections_table,
    sync_audio_batches,
)


def _sample_rows() -> list[dict[str, object]]:
    return [
        {
            "detection_key": "aaaaaaaaaaaaaaaa",
            "audio_id": "audio_1",
            "scientific_name": "Cyanocorax cyanopogon",
            "confidence": 0.91,
            "start_time": 0.0,
            "end_time": 1.2,
        },
        {
            "detection_key": "bbbbbbbbbbbbbbbb",
            "audio_id": "audio_2",
            "scientific_name": "Ramphastos toco",
            "confidence": 0.88,
            "start_time": 0.5,
            "end_time": 2.2,
        },
        {
            "detection_key": "cccccccccccccccc",
            "audio_id": "audio_2",
            "scientific_name": "Psarocolius decumanus",
            "confidence": 0.77,
            "start_time": 1.1,
            "end_time": 2.8,
        },
    ]


def test_load_detections_table_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "detections.csv"
    pd.DataFrame(_sample_rows()).to_csv(csv_path, index=False)

    loaded = load_detections_table(str(csv_path))

    assert len(loaded) == 3
    assert list(loaded.columns).count("detection_key") == 1


def test_load_detections_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame([{"detection_key": "aaaaaaaaaaaaaaaa", "audio_id": "audio_1"}]).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        _ = load_detections_table(str(csv_path))


def test_build_shards_in_directory(tmp_path: Path) -> None:
    frame = pd.DataFrame(_sample_rows())
    output_dir = tmp_path / "index" / "shards"

    shards = build_shards_in_directory(frame=frame, output_dir=output_dir, shard_size=2)

    assert len(shards) == 2
    assert (output_dir / "shard-00000.parquet").exists()
    assert (output_dir / "shard-00001.parquet").exists()
    assert shards[0].rows == 2
    assert shards[1].rows == 1


def test_manifest_payload_has_expected_stats() -> None:
    frame = pd.DataFrame(_sample_rows())
    payload = build_manifest_payload(
        project_slug="ppbio-rabeca",
        dataset_repo="org/birdnet-ppbio-rabeca-dataset",
        frame=frame,
        shard_size=1000,
        shard_metadata=[],
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["index"]["total_detections"] == 3
    assert payload["index"]["total_audio_files"] == 2


def test_collect_verify_errors_for_missing_prefix_and_shard() -> None:
    manifest_payload = {
        "schema_version": SCHEMA_VERSION,
        "project_slug": "ppbio-rabeca",
        "index": {
            "shards": [
                {
                    "path": "index/shards/shard-00000.parquet",
                    "rows": 2,
                    "sha256": "abc",
                    "size_bytes": 10,
                }
            ]
        },
    }

    repo_files = {
        "manifest.json",
        "audio/.gitkeep",
        "index/.gitkeep",
        "validations/.gitkeep",
        "audit/.gitkeep",
    }

    errors = collect_verify_errors(
        repo_files=repo_files,
        manifest_payload=manifest_payload,
        project_slug="ppbio-rabeca",
    )

    assert any("index/shards/" in message for message in errors)
    assert any("Shard referenced in manifest" in message for message in errors)


def test_manifest_json_roundtrip(tmp_path: Path) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "project_slug": "ppbio-rabeca",
        "index": {"shards": []},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["project_slug"] == "ppbio-rabeca"


def test_discover_audio_files_filters_supported_extensions(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "a.wav").write_bytes(b"wav")
    (audio_dir / "b.mp3").write_bytes(b"mp3")
    (audio_dir / "c.txt").write_text("ignore", encoding="utf-8")

    files = discover_audio_files(str(audio_dir))
    names = [item.name for item in files]

    assert names == ["a.wav", "b.mp3"]


def test_chunk_items() -> None:
    files = [Path(f"f{i}.wav") for i in range(5)]
    chunks = _chunk_items(files, 2)

    assert len(chunks) == 3
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2
    assert len(chunks[2]) == 1


def test_resume_state_roundtrip(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    _save_resume_state(state_file=state_file, uploaded={"audio/a.wav"}, failed={"audio/b.wav"})

    loaded = _load_resume_state(state_file=state_file)

    assert loaded["uploaded"] == ["audio/a.wav"]
    assert loaded["failed"] == ["audio/b.wav"]


def test_sync_audio_batches_skips_remote_and_checkpointed(tmp_path: Path) -> None:
    class FakeApi:
        def __init__(self) -> None:
            self.files = {
                "audio/existing.wav",
                "manifest.json",
                "audio/.gitkeep",
                "index/.gitkeep",
                "index/shards/.gitkeep",
                "validations/.gitkeep",
                "audit/.gitkeep",
            }
            self.uploaded: list[str] = []

        def create_repo(self, **kwargs: object) -> None:
            _ = kwargs

        def list_repo_files(self, **kwargs: object) -> list[str]:
            _ = kwargs
            return sorted(self.files)

        def upload_file(self, *, path_or_fileobj: object, path_in_repo: str, **kwargs: object) -> None:
            _ = path_or_fileobj
            _ = kwargs
            self.uploaded.append(path_in_repo)
            self.files.add(path_in_repo)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "existing.wav").write_bytes(b"x")
    (audio_dir / "new.wav").write_bytes(b"x")
    (audio_dir / "checkpointed.wav").write_bytes(b"x")

    state_file = tmp_path / "resume.json"
    state_file.write_text(
        json.dumps({"uploaded": ["audio/checkpointed.wav"], "failed": []}),
        encoding="utf-8",
    )

    api = FakeApi()
    result = sync_audio_batches(
        api=api,
        project_slug="ppbio-rabeca",
        dataset_repo="org/birdnet-ppbio-rabeca-dataset",
        local_audio_dir=str(audio_dir),
        batch_size=2,
        max_retries=1,
        retry_backoff_seconds=0,
        resume_state_file=str(state_file),
    )

    assert result["total_local_audio_files"] == 3
    assert result["pending_uploads"] == 1
    assert result["uploaded_now"] == 1
    assert result["failed"] == 0
    assert api.uploaded == ["audio/new.wav"]
