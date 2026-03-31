import json
from pathlib import Path

from src.cli.project_cli import main


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_create_project_writes_projects_and_owner_acl(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.json"
    access_file = tmp_path / "user_access.json"
    projects_file.write_text("[]\n", encoding="utf-8")
    access_file.write_text("{}\n", encoding="utf-8")

    exit_code = main(
        [
            "create-project",
            "--projects-file",
            str(projects_file),
            "--user-access-file",
            str(access_file),
            "--slug",
            "amazonia-2026",
            "--name",
            "Amazonia 2026",
            "--dataset-repo-id",
            "birdnet/amazonia-2026",
            "--owner",
            "admin_user",
        ]
    )

    assert exit_code == 0
    projects = _read_json(projects_file)
    assert len(projects) == 1
    assert projects[0]["project_slug"] == "amazonia-2026"
    assert projects[0]["visibility"] == "collaborative"

    access = _read_json(access_file)
    assert access["admin_user"]["amazonia-2026"] == "admin"


def test_create_project_duplicate_slug_returns_error(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.json"
    projects_file.write_text(
        json.dumps(
            [
                {
                    "project_slug": "p1",
                    "name": "Project 1",
                    "dataset_repo_id": "org/p1",
                    "active": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "create-project",
            "--projects-file",
            str(projects_file),
            "--slug",
            "p1",
            "--name",
            "Project 1",
            "--dataset-repo-id",
            "org/p1",
        ]
    )

    assert exit_code == 1


def test_init_build_and_verify_project_flow(tmp_path: Path) -> None:
    projects_file = tmp_path / "projects.json"
    projects_file.write_text(
        json.dumps(
            [
                {
                    "project_slug": "p2",
                    "name": "Project 2",
                    "dataset_repo_id": "org/p2",
                    "active": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    dataset_root = tmp_path / "datasets"

    init_exit = main(
        [
            "init-dataset",
            "--dataset-root",
            str(dataset_root),
            "--slug",
            "p2",
            "--dataset-repo-id",
            "org/p2",
            "--name",
            "Project 2",
        ]
    )
    assert init_exit == 0

    detections_file = dataset_root / "p2" / "detections" / "detections.jsonl"
    detections_file.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "detection_key": "k1",
                        "audio_id": "a1",
                        "scientific_name": "Sp A",
                        "confidence": 0.6,
                    }
                ),
                json.dumps(
                    {
                        "detection_key": "k2",
                        "audio_id": "a2",
                        "scientific_name": "Sp B",
                        "confidence": 0.9,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    build_exit = main(
        [
            "build-index",
            "--dataset-root",
            str(dataset_root),
            "--slug",
            "p2",
        ]
    )
    assert build_exit == 0

    index_payload = _read_json(dataset_root / "p2" / "index" / "detections_index.json")
    assert index_payload["count"] == 2
    assert index_payload["detections"][0]["detection_key"] == "k2"

    verify_exit = main(
        [
            "verify-project",
            "--projects-file",
            str(projects_file),
            "--dataset-root",
            str(dataset_root),
            "--slug",
            "p2",
        ]
    )
    assert verify_exit == 0
