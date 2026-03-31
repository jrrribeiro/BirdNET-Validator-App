from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIRS = [
    "audio",
    "detections",
    "index",
    "exports",
]


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _project_exists(projects: list[dict[str, Any]], slug: str) -> bool:
    return any(p.get("project_slug") == slug for p in projects)


def cmd_create_project(args: argparse.Namespace) -> int:
    projects_file = Path(args.projects_file)
    projects: list[dict[str, Any]] = _load_json(projects_file, [])

    if _project_exists(projects, args.slug):
        print(f"ERROR: project '{args.slug}' already exists")
        return 1

    project_entry: dict[str, Any] = {
        "project_slug": args.slug,
        "name": args.name,
        "dataset_repo_id": args.dataset_repo_id,
        "visibility": args.visibility,
        "active": not args.inactive,
    }
    if args.owner:
        project_entry["owner_username"] = args.owner
    if args.dataset_token:
        project_entry["dataset_token"] = args.dataset_token

    projects.append(project_entry)
    projects.sort(key=lambda p: str(p.get("project_slug", "")))
    _write_json(projects_file, projects)

    if args.user_access_file and args.owner:
        access_file = Path(args.user_access_file)
        access_payload: dict[str, dict[str, str]] = _load_json(access_file, {})
        owner_roles = access_payload.setdefault(args.owner, {})
        owner_roles[args.slug] = "admin"
        _write_json(access_file, access_payload)

    print(f"OK: project '{args.slug}' created")
    return 0


def cmd_init_dataset(args: argparse.Namespace) -> int:
    root = Path(args.dataset_root) / args.slug
    for dirname in DEFAULT_DATASET_DIRS:
        (root / dirname).mkdir(parents=True, exist_ok=True)

    detections_file = root / "detections" / "detections.jsonl"
    if not detections_file.exists():
        detections_file.write_text("", encoding="utf-8")

    metadata = {
        "project_slug": args.slug,
        "dataset_repo_id": args.dataset_repo_id,
    }
    if args.name:
        metadata["name"] = args.name

    _write_json(root / "index" / "project_metadata.json", metadata)
    print(f"OK: dataset scaffold initialized at {root}")
    return 0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return rows


def cmd_build_index(args: argparse.Namespace) -> int:
    project_root = Path(args.dataset_root) / args.slug
    detections_file = project_root / "detections" / "detections.jsonl"
    rows = _read_jsonl(detections_file)

    sorted_rows = sorted(rows, key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
    index_payload = {
        "project_slug": args.slug,
        "count": len(sorted_rows),
        "detections": [
            {
                "detection_key": r.get("detection_key"),
                "audio_id": r.get("audio_id"),
                "scientific_name": r.get("scientific_name"),
                "confidence": float(r.get("confidence", 0.0)),
            }
            for r in sorted_rows
        ],
    }

    _write_json(project_root / "index" / "detections_index.json", index_payload)
    print(f"OK: index generated with {len(sorted_rows)} detections")
    return 0


def cmd_verify_project(args: argparse.Namespace) -> int:
    projects_file = Path(args.projects_file)
    projects: list[dict[str, Any]] = _load_json(projects_file, [])

    if not _project_exists(projects, args.slug):
        print(f"ERROR: project '{args.slug}' not found in {projects_file}")
        return 1

    project_root = Path(args.dataset_root) / args.slug
    missing: list[str] = []

    for dirname in DEFAULT_DATASET_DIRS:
        expected = project_root / dirname
        if not expected.exists():
            missing.append(str(expected))

    required_files = [
        project_root / "detections" / "detections.jsonl",
        project_root / "index" / "project_metadata.json",
        project_root / "index" / "detections_index.json",
    ]
    for required in required_files:
        if not required.exists():
            missing.append(str(required))

    if missing:
        print("ERROR: project verification failed; missing paths:")
        for item in missing:
            print(f" - {item}")
        return 1

    print(f"OK: project '{args.slug}' verified")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="birdnet-project")
    sub = parser.add_subparsers(dest="command", required=True)

    create_project = sub.add_parser("create-project", help="Create project in bootstrap JSON")
    create_project.add_argument("--projects-file", required=True)
    create_project.add_argument("--user-access-file")
    create_project.add_argument("--slug", required=True)
    create_project.add_argument("--name", required=True)
    create_project.add_argument("--dataset-repo-id", required=True)
    create_project.add_argument("--visibility", choices=["private", "collaborative"], default="collaborative")
    create_project.add_argument("--owner")
    create_project.add_argument("--dataset-token")
    create_project.add_argument("--inactive", action="store_true")
    create_project.set_defaults(func=cmd_create_project)

    init_dataset = sub.add_parser("init-dataset", help="Create initial dataset folder scaffold")
    init_dataset.add_argument("--dataset-root", required=True)
    init_dataset.add_argument("--slug", required=True)
    init_dataset.add_argument("--dataset-repo-id", required=True)
    init_dataset.add_argument("--name")
    init_dataset.set_defaults(func=cmd_init_dataset)

    build_index = sub.add_parser("build-index", help="Build simple index from detections JSONL")
    build_index.add_argument("--dataset-root", required=True)
    build_index.add_argument("--slug", required=True)
    build_index.set_defaults(func=cmd_build_index)

    verify_project = sub.add_parser("verify-project", help="Verify project config and local scaffold")
    verify_project.add_argument("--projects-file", required=True)
    verify_project.add_argument("--dataset-root", required=True)
    verify_project.add_argument("--slug", required=True)
    verify_project.set_defaults(func=cmd_verify_project)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
