import json
from pathlib import Path

import pandas as pd
import pytest

from cli.hf_dataset_cli import (
    SCHEMA_VERSION,
    build_manifest_payload,
    build_shards_in_directory,
    collect_verify_errors,
    load_detections_table,
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
