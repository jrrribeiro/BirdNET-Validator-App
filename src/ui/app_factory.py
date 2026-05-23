import gradio as gr
import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
import wave
from collections.abc import Mapping
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from src.config.runtime_config import RuntimeConfig
from src.cache.ephemeral_cache_manager import EphemeralCacheManager
from src.domain.models import Detection, Project, Role
from src.repositories.append_only_validation_repository import AppendOnlyValidationRepository, OptimisticLockError
from src.repositories.in_memory_detection_repository import InMemoryDetectionRepository
from src.repositories.supabase_state import SupabaseBootstrapStore, SupabaseRestClient, SupabaseStateError, SupabaseValidationRepository
from src.services.audio_fetch_service import AudioFetchService
from src.services.detection_queue_service import DetectionQueueService
from src.services.validation_service import ValidationService
from src.services.invite_email_notifier import EmailJSInviteEmailNotifier, InviteEmailNotifier
from src.auth.auth_service import AuthService
from src.ui.login_page import create_login_page
from src.ui.admin_panel import AdminPanelManager
from src.ui.components import admin_overview_html, compact_metric_grid, coverage_bars_html, inline_hint_html, invite_panel_html, paged_activity_html, project_context_html, project_overview_html, section_header_html, selected_segment_html, settings_health_html
from src.ui.theme import APP_CSS, app_header_html


class _AudioFetchResultProtocol(Protocol):
    cache_key: str
    local_path: str
    source: str


class _AudioServiceProtocol(Protocol):
    def fetch(
        self,
        dataset_repo: str,
        audio_id: str,
        allow_demo_fallback: bool = False,
        hf_token: str | None = None,
    ) -> _AudioFetchResultProtocol: ...

    def cleanup_after_validation(self, cache_key: str) -> None: ...


class _ValidationServiceProtocol(Protocol):
    def validate_detection(
        self,
        project_slug: str,
        detection_key: str,
        status: str,
        validator: str,
        notes: str = "",
        corrected_species: str | None = None,
        expected_version: int | None = None,
    ) -> object: ...


class _ValidationReadRepositoryProtocol(Protocol):
    def load_current_snapshot(self, project_slug: str) -> dict[str, dict[str, object]]: ...

    def list_events(self, project_slug: str) -> list[dict[str, object]]: ...


class _QueueServiceProtocol(Protocol):
    def list_all_detections(
        self,
        project_slug: str,
        scientific_name: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
    ) -> list[Detection]: ...

    def get_page(
        self,
        project_slug: str,
        page: int,
        page_size: int,
        scientific_name: str | None = None,
        min_confidence: float | None = None,
        max_confidence: float | None = None,
    ) -> object: ...


def _seed_service() -> DetectionQueueService:
    return _seed_service_for_projects(["demo-project"])[0]


def _candidate_metadata_files(project_slug: str) -> list[str]:
    return [
        f"{project_slug}/detections.jsonl",
        f"{project_slug}/detections.json",
        f"{project_slug}/detections.csv",
        f"{project_slug}/segments.jsonl",
        f"{project_slug}/segments.json",
        f"{project_slug}/segments.csv",
        "detections.jsonl",
        "detections.json",
        "detections.csv",
        "segments.jsonl",
        "segments.json",
        "segments.csv",
        "metadata/detections.jsonl",
        "metadata/detections.json",
        "metadata/detections.csv",
        "metadata/segments.jsonl",
        "metadata/segments.json",
        "metadata/segments.csv",
        "validation/detections.jsonl",
        "validation/detections.json",
        "validation/detections.csv",
    ]


def _pick_row_value(raw: dict[str, object], keys: list[str]) -> str:
    for key in keys:
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _to_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_audio_id(audio_value: str) -> str:
    normalized = audio_value.strip().replace("\\", "/")
    if normalized.startswith("audio/"):
        normalized = normalized[len("audio/") :]
    return normalized


def _build_detection_from_row(raw: dict[str, object], row_index: int, project_slug: str) -> Detection | None:
    row_project = _pick_row_value(raw, ["project_slug", "project", "project_id"])
    if row_project and row_project != project_slug:
        return None

    audio_id = _normalize_audio_id(
        _pick_row_value(
            raw,
            [
                "segment_path_in_repo",
                "stored_path",
                "segment_relpath",
                "audio_id",
                "audio_file",
                "audio_path",
                "segment_path",
                "file",
                "filepath",
                "path",
                "filename",
            ],
        )
    )
    if not audio_id:
        return None

    scientific_name = _pick_row_value(
        raw,
        [
            "scientific_name",
            "species",
            "species_name",
            "predicted_species",
            "logical_group",
            "label",
            "taxon",
        ],
    )
    if not scientific_name:
        scientific_name = "Unknown species"

    confidence = _to_float(
        raw.get("confidence", raw.get("score", raw.get("probability", raw.get("prediction_confidence", 1.0)))),
        1.0,
    )
    confidence = max(0.0, min(1.0, confidence))

    start_time = _to_float(
        raw.get("start_time", raw.get("start", raw.get("begin", raw.get("offset", raw.get("segment_start", 0.0))))),
        0.0,
    )
    end_time = _to_float(
        raw.get("end_time", raw.get("end", raw.get("stop", raw.get("segment_end", 0.0)))),
        0.0,
    )
    if end_time <= 0.0:
        duration = _to_float(raw.get("duration", 0.0), 0.0)
        if duration > 0.0:
            end_time = start_time + duration
    if end_time <= start_time:
        end_time = start_time + 1.0

    detection_key = _pick_row_value(raw, ["detection_key", "segment_id", "id", "uid", "key"])
    if not detection_key:
        stable = f"{project_slug}|{audio_id}|{scientific_name}|{start_time:.3f}|{end_time:.3f}|{row_index}"
        detection_key = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:16]

    try:
        return Detection(
            detection_key=detection_key,
            audio_id=audio_id,
            scientific_name=scientific_name,
            confidence=confidence,
            start_time=start_time,
            end_time=end_time,
            source_metadata=dict(raw),
        )
    except Exception:
        return None


def _parse_detection_metadata_payload(payload: object, project_slug: str) -> list[Detection]:
    rows: list[dict[str, object]] = []
    if isinstance(payload, list):
        rows = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        project_rows = payload.get(project_slug)
        if isinstance(project_rows, list):
            rows = [item for item in project_rows if isinstance(item, dict)]
        else:
            for key in ["detections", "segments", "items", "rows"]:
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    rows = [item for item in candidate if isinstance(item, dict)]
                    break

    parsed: list[Detection] = []
    seen_keys: set[str] = set()
    for index, row in enumerate(rows):
        detection = _build_detection_from_row(row, index, project_slug)
        if detection is None:
            continue
        if detection.detection_key in seen_keys:
            continue
        seen_keys.add(detection.detection_key)
        parsed.append(detection)
    return parsed


def _env_hf_token() -> str | None:
    token = (
        os.getenv("BIRDNET_HF_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or ""
    ).strip()
    return token or None


def _project_dataset_token(project: Project, fallback_token: str | None = None) -> str | None:
    return (project.dataset_token or "").strip() or (fallback_token or "").strip() or _env_hf_token()


def _resolve_project_fetch_token(project: Project | None, session_token: str | None = None) -> str | None:
    return (session_token or "").strip() or ((project.dataset_token or "").strip() if project is not None else "") or _env_hf_token()


def _load_dataset_detections_for_project(project: Project, hf_token: str | None = None) -> tuple[list[Detection], str]:
    dataset_repo = project.dataset_repo_id.strip()
    if not dataset_repo:
        return [], ""

    token = _project_dataset_token(project, hf_token)
    fast_index_detections, fast_index_warning = _load_detections_from_known_files_index(
        project=project,
        dataset_repo=dataset_repo,
        token=token,
    )
    if fast_index_detections:
        return fast_index_detections, fast_index_warning

    try:
        api = HfApi(token=token)
        repo_files = api.list_repo_files(repo_id=dataset_repo, repo_type="dataset")
    except Exception as exc:
        if _is_hf_rate_limit_error(exc):
            token_hint = (
                " Login with a Hugging Face token or store a project token with dataset read access."
                if token is None
                else " The current token was still rate-limited by Hugging Face."
            )
            return [], (
                f"⚠️ Hugging Face rate-limited dataset discovery for {dataset_repo}: {exc}\n\n"
                "The app first tried the fast HF_Dataset_Uploader index path, but it was unavailable. "
                "Legacy dataset discovery falls back to the Hub tree API, which has stricter limits."
                f"{token_hint} Retry after the Hub rate-limit window resets."
            )
        token_hint = (
            " No Hugging Face token is configured for this project/session."
            if token is None
            else " The configured token could not access this dataset."
        )
        return [], (
            f"⚠️ Could not list files for dataset {dataset_repo}: {exc}\n\n"
            f"{token_hint} If the dataset is private or gated, add a project token in Admin > Project token management "
            "or login with a Hugging Face token that has dataset read access."
        )

    if not repo_files:
        return [], f"⚠️ Dataset {dataset_repo} has no files."

    files_index_detections, files_index_warning = _load_detections_from_files_index(
        project=project,
        dataset_repo=dataset_repo,
        token=token,
        repo_files=repo_files,
    )
    if files_index_detections:
        return files_index_detections, files_index_warning

    shard_detections, shard_warning = _load_detections_from_parquet_shards(
        project=project,
        dataset_repo=dataset_repo,
        token=token,
        repo_files=repo_files,
    )
    if shard_detections:
        return shard_detections, shard_warning

    preferred = _candidate_metadata_files(project.project_slug)
    selected_file = next((name for name in preferred if name in repo_files), "")
    if not selected_file:
        metadata_candidates = []
        for name in repo_files:
            lowered = name.lower()
            if lowered.startswith("audio/"):
                continue
            if not lowered.endswith((".json", ".jsonl", ".csv")):
                continue
            if "detection" in lowered or "segment" in lowered:
                metadata_candidates.append(name)
        if metadata_candidates:
            selected_file = sorted(metadata_candidates, key=lambda value: (len(value), value))[0]

    if not selected_file:
        parsed_from_paths = _build_detections_from_audio_paths(project, repo_files)
        if parsed_from_paths:
            return parsed_from_paths, ""
        return [], (
            f"⚠️ Dataset {dataset_repo} has no detection metadata file for project {project.project_slug}. "
            "Expected names like detections.jsonl / segments.csv (top-level, metadata/, or <project_slug>/), "
            "index/files.parquet from HF_Dataset_Uploader, or audio files under audio/segments/<species>/..."
        )

    try:
        if token:
            downloaded_path = hf_hub_download(
                repo_id=dataset_repo,
                repo_type="dataset",
                filename=selected_file,
                token=token,
            )
        else:
            downloaded_path = hf_hub_download(
                repo_id=dataset_repo,
                repo_type="dataset",
                filename=selected_file,
            )
    except Exception as exc:
        return [], f"⚠️ Failed to download {selected_file} from {dataset_repo}: {exc}"

    metadata_path = Path(downloaded_path)
    try:
        if metadata_path.suffix.lower() == ".jsonl":
            rows = []
            for line in metadata_path.read_text(encoding="utf-8").splitlines():
                text = line.strip()
                if not text:
                    continue
                value = json.loads(text)
                if isinstance(value, dict):
                    rows.append(value)
            parsed = _parse_detection_metadata_payload(rows, project.project_slug)
        elif metadata_path.suffix.lower() == ".csv":
            with metadata_path.open("r", encoding="utf-8", newline="") as file_handle:
                parsed = _parse_detection_metadata_payload(list(csv.DictReader(file_handle)), project.project_slug)
        else:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            parsed = _parse_detection_metadata_payload(payload, project.project_slug)
    except Exception as exc:
        return [], f"⚠️ Failed to parse detection metadata {selected_file} from {dataset_repo}: {exc}"

    if not parsed:
        parsed_from_paths = _build_detections_from_audio_paths(project, repo_files)
        if parsed_from_paths:
            return parsed_from_paths, ""
        return [], (
            f"⚠️ Metadata file {selected_file} from {dataset_repo} has no valid detections for project {project.project_slug}."
        )

    return parsed, ""


def _is_hf_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "429" in message or "too many requests" in message or "rate limit" in message


def _load_detections_from_known_files_index(
    project: Project,
    dataset_repo: str,
    token: str | None,
) -> tuple[list[Detection], str]:
    """Load the uploader index through direct file downloads before repo tree discovery."""
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        return [], ""

    index_warning = ""
    for files_index_path in ["index/files.parquet", "files.parquet", "metadata/files.parquet"]:
        try:
            downloaded = _download_dataset_metadata_file(dataset_repo, files_index_path, token)
            frame = pd.read_parquet(downloaded)
        except Exception:
            continue

        rows = frame.to_dict(orient="records")
        rows = _try_merge_known_detection_csv(dataset_repo, token, rows, pd)
        parsed = _parse_detection_metadata_payload(rows, project.project_slug)
        if parsed:
            return parsed, ""

        index_warning = (
            f"⚠️ Dataset {dataset_repo} has {files_index_path}, but no valid audio rows were parsed "
            f"for project {project.project_slug}."
        )

    return [], index_warning


def _try_merge_known_detection_csv(
    dataset_repo: str,
    token: str | None,
    file_rows: list[dict[str, object]],
    pandas_module,
) -> list[dict[str, object]]:
    for detections_csv_path in ["index/detections.csv", "detections.csv", "metadata/detections.csv"]:
        try:
            csv_path = _download_dataset_metadata_file(dataset_repo, detections_csv_path, token)
            detections_frame = pandas_module.read_csv(csv_path)
            return _merge_files_index_with_detection_rows(file_rows, detections_frame.to_dict(orient="records"))
        except Exception:
            continue
    return file_rows


def _load_detections_from_files_index(
    project: Project,
    dataset_repo: str,
    token: str | None,
    repo_files: list[str],
) -> tuple[list[Detection], str]:
    files_index_path = _resolve_files_index_path(repo_files)
    if not files_index_path:
        return [], ""

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        return [], (
            f"⚠️ Dataset {dataset_repo} contains {files_index_path}, but pandas/pyarrow are unavailable to read it."
        )

    try:
        downloaded = _download_dataset_metadata_file(dataset_repo, files_index_path, token)
        frame = pd.read_parquet(downloaded)
        rows = frame.to_dict(orient="records")
    except Exception as exc:
        return [], f"⚠️ Failed to read {files_index_path} from {dataset_repo}: {exc}"

    detections_csv_path = _resolve_detection_csv_path(repo_files)
    if detections_csv_path:
        try:
            csv_path = _download_dataset_metadata_file(dataset_repo, detections_csv_path, token)
            detections_frame = pd.read_csv(csv_path)
            rows = _merge_files_index_with_detection_rows(rows, detections_frame.to_dict(orient="records"))
        except Exception:
            # The files index is enough to validate audio segments. CSV enrichment is optional.
            pass

    parsed = _parse_detection_metadata_payload(rows, project.project_slug)
    if not parsed:
        return [], f"⚠️ Dataset {dataset_repo} has {files_index_path}, but no valid audio rows were parsed."
    return parsed, ""


def _resolve_files_index_path(repo_files: list[str]) -> str:
    preferred = ["index/files.parquet", "files.parquet", "metadata/files.parquet"]
    for candidate in preferred:
        if candidate in repo_files:
            return candidate
    for path in repo_files:
        normalized = str(path).replace("\\", "/").strip().lower()
        if normalized.endswith("/files.parquet") or normalized == "files.parquet":
            return str(path)
    return ""


def _resolve_detection_csv_path(repo_files: list[str]) -> str:
    preferred = ["index/detections.csv", "detections.csv", "metadata/detections.csv"]
    for candidate in preferred:
        if candidate in repo_files:
            return candidate
    return ""


def _download_dataset_metadata_file(dataset_repo: str, filename: str, token: str | None) -> str:
    if token:
        return hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            filename=filename,
            token=token,
        )
    return hf_hub_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        filename=filename,
    )


def _merge_files_index_with_detection_rows(
    file_rows: list[dict[str, object]],
    detection_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    detection_by_species_and_source: dict[tuple[str, str], dict[str, object]] = {}
    detection_by_source: dict[str, dict[str, object]] = {}

    for row in detection_rows:
        source_file = _pick_row_value(row, ["source_file", "audio_file", "file", "filename", "path"])
        species = _pick_row_value(row, ["scientific_name", "species", "label", "logical_group"])
        if source_file:
            detection_by_source.setdefault(source_file, row)
            if species:
                detection_by_species_and_source.setdefault((species, source_file), row)

    merged: list[dict[str, object]] = []
    for row in file_rows:
        combined = dict(row)
        original_relative_path = _pick_row_value(row, ["original_relative_path", "relative_path", "stored_path", "audio_id"])
        filename = _pick_row_value(row, ["filename"])
        if not filename and original_relative_path:
            filename = Path(original_relative_path).name
        species = _pick_row_value(row, ["logical_group", "scientific_name", "species"])

        match = None
        if species and filename:
            match = detection_by_species_and_source.get((species, filename))
        if match is None and filename:
            match = detection_by_source.get(filename)
        if match:
            for key, value in match.items():
                combined.setdefault(key, value)
        merged.append(combined)
    return merged


def _resolve_shard_paths_from_repo_files(repo_files: list[str]) -> list[str]:
    return sorted(
        {
            file_path
            for file_path in repo_files
            if str(file_path).lower().startswith("index/shards/") and str(file_path).lower().endswith(".parquet")
        }
    )


def _load_detections_from_parquet_shards(
    project: Project,
    dataset_repo: str,
    token: str | None,
    repo_files: list[str],
) -> tuple[list[Detection], str]:
    shard_paths = _resolve_shard_paths_from_repo_files(repo_files)

    if "manifest.json" in repo_files:
        try:
            if token:
                manifest_path = hf_hub_download(
                    repo_id=dataset_repo,
                    repo_type="dataset",
                    filename="manifest.json",
                    token=token,
                )
            else:
                manifest_path = hf_hub_download(
                    repo_id=dataset_repo,
                    repo_type="dataset",
                    filename="manifest.json",
                )
            manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            manifest_shards = manifest_payload.get("index", {}).get("shards", [])
            if isinstance(manifest_shards, list):
                manifest_paths = [
                    str(item.get("path", "")).strip()
                    for item in manifest_shards
                    if isinstance(item, dict)
                ]
                manifest_paths = [p for p in manifest_paths if p.lower().endswith(".parquet")]
                if manifest_paths:
                    shard_paths = manifest_paths
        except Exception:
            pass

    if not shard_paths:
        return [], ""

    try:
        import pandas as pd  # type: ignore[import-not-found]
    except Exception:
        return [], (
            f"⚠️ Dataset {dataset_repo} contains parquet index shards, but pandas/pyarrow are unavailable to read them."
        )

    rows: list[dict[str, object]] = []
    for shard_path in shard_paths:
        try:
            if token:
                downloaded = hf_hub_download(
                    repo_id=dataset_repo,
                    repo_type="dataset",
                    filename=shard_path,
                    token=token,
                )
            else:
                downloaded = hf_hub_download(
                    repo_id=dataset_repo,
                    repo_type="dataset",
                    filename=shard_path,
                )
            frame = pd.read_parquet(downloaded)
            rows.extend(frame.to_dict(orient="records"))
        except Exception:
            continue

    parsed = _parse_detection_metadata_payload(rows, project.project_slug)
    if not parsed:
        return [], f"⚠️ Dataset {dataset_repo} index shards were found but contain no rows for project {project.project_slug}."
    return parsed, ""


def _parse_segment_filename_hint(filename: str) -> tuple[float, float, float]:
    # Common uploader pattern: ..._12.0-15.0s_85%.wav
    segment_match = re.search(r"_(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)s_(\d+(?:\.\d+)?)%", filename)
    if segment_match:
        start_time = float(segment_match.group(1))
        end_time = float(segment_match.group(2))
        confidence = float(segment_match.group(3)) / 100.0
        return start_time, end_time, max(0.0, min(1.0, confidence))

    # Fallback pattern without confidence: ..._12.0-15.0s
    basic_match = re.search(r"_(\d+(?:\.\d+)?)\-(\d+(?:\.\d+)?)s", filename)
    if basic_match:
        return float(basic_match.group(1)), float(basic_match.group(2)), 0.5

    return 0.0, 1.0, 0.5


def _build_detections_from_audio_paths(project: Project, repo_files: list[str]) -> list[Detection]:
    detections: list[Detection] = []
    seen_keys: set[str] = set()

    for file_path in repo_files:
        normalized = str(file_path).replace("\\", "/").strip()
        lower = normalized.lower()
        if not lower.startswith("audio/"):
            continue
        if not lower.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            continue

        relative_audio_id = normalized[len("audio/") :]
        parts = relative_audio_id.split("/")
        if len(parts) < 2:
            continue

        if parts[0].lower() == "segments" and len(parts) >= 3:
            scientific_name = parts[1].replace("_", " ").strip() or "Unknown species"
        else:
            scientific_name = parts[-2].replace("_", " ").strip() or "Unknown species"

        filename = parts[-1]
        start_time, end_time, confidence = _parse_segment_filename_hint(filename)
        stable = f"{project.project_slug}|{relative_audio_id}|{scientific_name}|{start_time:.3f}|{end_time:.3f}"
        detection_key = hashlib.sha1(stable.encode("utf-8")).hexdigest()[:16]
        if detection_key in seen_keys:
            continue

        try:
            detections.append(
                Detection(
                    detection_key=detection_key,
                    audio_id=relative_audio_id,
                    scientific_name=scientific_name,
                    confidence=confidence,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
            seen_keys.add(detection_key)
        except Exception:
            continue

    return detections


def _seed_service_for_projects(
    project_slugs: list[str],
    seed_file_path: str | None = None,
    project_map: dict[str, Project] | None = None,
    allow_demo_defaults: bool = True,
    hf_token: str | None = None,
) -> tuple[DetectionQueueService, list[str]]:
    repo = InMemoryDetectionRepository()
    detected_by_project = _load_seed_detections(seed_file_path)
    warnings: list[str] = []

    for project_slug in project_slugs:
        items, dataset_warning = _project_detection_items(
            project_slug,
            seed_detections_by_project=detected_by_project,
            project_map=project_map,
            allow_demo_defaults=allow_demo_defaults,
            hf_token=hf_token,
        )
        if dataset_warning:
            warnings.append(dataset_warning)
        repo.seed(project_slug, items)

    return DetectionQueueService(repo), warnings


def _build_detection_repository(
    project_slugs: list[str],
    seed_file_path: str | None,
    project_map: dict[str, Project] | None = None,
    allow_demo_defaults: bool = True,
    hf_token: str | None = None,
) -> tuple[DetectionQueueService, str]:
    warning = _validate_seed_file(seed_file_path)
    service, dataset_warnings = _seed_service_for_projects(
        project_slugs,
        seed_file_path=seed_file_path,
        project_map=project_map,
        allow_demo_defaults=allow_demo_defaults,
        hf_token=hf_token,
    )

    warnings = [item for item in [warning, *dataset_warnings] if item.strip()]
    joined_warning = "\n\n".join(dict.fromkeys(warnings))
    return service, joined_warning


def _validate_seed_file(seed_file_path: str | None) -> str:
    if not seed_file_path:
        return ""

    normalized_path = Path(seed_file_path)
    if not normalized_path.exists():
        return (
            f"⚠️ BIRDNET_DETECTIONS_FILE not found: {normalized_path}. "
            "Set BIRDNET_DETECTIONS_FILE to a valid JSON file path or unset it to use default demo detections."
        )

    try:
        payload = json.loads(normalized_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return (
            f"⚠️ BIRDNET_DETECTIONS_FILE invalid: {exc}. "
            "Fix JSON syntax and ensure UTF-8 encoding."
        )

    if isinstance(payload, dict):
        non_list_projects = [slug for slug, rows in payload.items() if not isinstance(rows, list)]
        if non_list_projects:
            sample = ", ".join(non_list_projects[:3])
            return (
                f"⚠️ Invalid seed JSON: projects without a detection list ({sample}). "
                "Each project key must map to a list of detection objects."
            )
        return ""

    if isinstance(payload, list):
        missing_project = 0
        for row in payload:
            if not isinstance(row, dict):
                continue
            if not str(row.get("project_slug", "")).strip():
                missing_project += 1
        if missing_project:
            return (
                "⚠️ Invalid seed JSON: entries without project_slug in list. "
                "Add project_slug to each detection object when using list format."
            )
        return ""

    return (
        "⚠️ Invalid seed JSON: format must be object-by-project or detection list. "
        "See README for supported examples."
    )


def _default_demo_detections(project_slug: str) -> list[Detection]:
    stable_prefix = hashlib.sha1(project_slug.encode("utf-8")).hexdigest()[:8]
    slug_prefix = project_slug.replace("-", "_")
    return [
        Detection(
            detection_key=f"{stable_prefix}00001001",
            audio_id=f"{slug_prefix}_audio_1001",
            scientific_name="Cyanocorax cyanopogon",
            confidence=0.93,
            start_time=1.2,
            end_time=2.5,
        ),
        Detection(
            detection_key=f"{stable_prefix}00001002",
            audio_id=f"{slug_prefix}_audio_1002",
            scientific_name="Ramphastos toco",
            confidence=0.88,
            start_time=0.8,
            end_time=2.1,
        ),
        Detection(
            detection_key=f"{stable_prefix}00001003",
            audio_id=f"{slug_prefix}_audio_1003",
            scientific_name="Cyanocorax cyanopogon",
            confidence=0.72,
            start_time=3.1,
            end_time=4.0,
        ),
        Detection(
            detection_key=f"{stable_prefix}00001004",
            audio_id=f"{slug_prefix}_audio_1004",
            scientific_name="Psarocolius decumanus",
            confidence=0.67,
            start_time=5.0,
            end_time=6.3,
        ),
    ]


def _load_seed_detections(seed_file_path: str | None) -> dict[str, list[Detection]]:
    if not seed_file_path:
        return {}

    normalized_path = Path(seed_file_path)
    if not normalized_path.exists():
        return {}

    try:
        payload = json.loads(normalized_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    result: dict[str, list[Detection]] = {}

    if isinstance(payload, dict):
        for project_slug, rows in payload.items():
            parsed_rows = _parse_detection_rows(rows)
            if parsed_rows:
                result[str(project_slug)] = parsed_rows
        return result

    if isinstance(payload, list):
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            project_slug = str(row.get("project_slug", "")).strip()
            if not project_slug:
                continue
            grouped.setdefault(project_slug, []).append(row)

        for project_slug, rows in grouped.items():
            parsed_rows = _parse_detection_rows(rows)
            if parsed_rows:
                result[project_slug] = parsed_rows

    return result


def _parse_detection_rows(rows: object) -> list[Detection]:
    parsed: list[Detection] = []
    if not isinstance(rows, list):
        return parsed

    for raw in rows:
        if not isinstance(raw, dict):
            continue
        try:
            parsed.append(
                Detection(
                    detection_key=str(raw.get("detection_key", "")).strip(),
                    audio_id=str(raw.get("audio_id", "")).strip(),
                    scientific_name=str(raw.get("scientific_name", "")).strip(),
                    confidence=float(raw.get("confidence", 0.0)),
                    start_time=float(raw.get("start_time", 0.0)),
                    end_time=float(raw.get("end_time", 0.0)),
                )
            )
        except Exception:
            continue

    return parsed


def _project_detection_items(
    project_slug: str,
    *,
    seed_detections_by_project: dict[str, list[Detection]],
    project_map: dict[str, Project] | None = None,
    allow_demo_defaults: bool = True,
    hf_token: str | None = None,
) -> tuple[list[Detection], str]:
    project = (project_map or {}).get(project_slug)
    dataset_items: list[Detection] = []
    dataset_warning = ""
    if project is not None and project.active:
        dataset_items, dataset_warning = _load_dataset_detections_for_project(project, hf_token=hf_token)

    seeded_items = seed_detections_by_project.get(project_slug, [])
    if dataset_items:
        items = dataset_items
    elif seeded_items:
        items = seeded_items
    elif allow_demo_defaults:
        items = _default_demo_detections(project_slug)
    else:
        items = []

    return sorted(items, key=lambda item: item.detection_key), dataset_warning


def _default_projects() -> list[Project]:
    return [
        Project(
            project_slug="kenya-2024",
            name="Kenya Survey 2024",
            dataset_repo_id="birdnet/kenya-2024-dataset",
            active=True,
        ),
        Project(
            project_slug="nairobi-2023",
            name="Nairobi Survey 2023",
            dataset_repo_id="birdnet/nairobi-2023-dataset",
            active=True,
        ),
        Project(
            project_slug="demo-project",
            name="Demo Project",
            dataset_repo_id="birdnet/demo-dataset",
            active=True,
        ),
    ]


def _default_user_access() -> dict[str, dict[str, Role]]:
    return {
        "demo_user": {"demo-project": Role.validator, "birds-local": Role.validator},
        "admin_user": {"kenya-2024": Role.admin, "nairobi-2023": Role.admin},
        "validator_demo": {"demo-project": Role.validator, "kenya-2024": Role.validator},
        "validator_other": {"nairobi-2023": Role.validator},
    }


def _load_projects_from_file(projects_file_path: str | None) -> list[Project]:
    if not projects_file_path:
        return []

    path = Path(projects_file_path)
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    projects: list[Project] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        try:
            slug = str(row.get("project_slug", "")).strip()
            project_id = str(row.get("project_id", "")).strip()
            if not project_id and slug:
                # Deterministic legacy migration so IDs are stable before first re-persist.
                project_id = str(uuid5(NAMESPACE_URL, f"birdnet-validator:{slug}"))
            projects.append(
                Project(
                    project_id=project_id or str(uuid4()),
                    project_slug=slug,
                    name=str(row.get("name", "")).strip(),
                    dataset_repo_id=str(row.get("dataset_repo_id", "")).strip(),
                    visibility=str(row.get("visibility", "collaborative")).strip() or "collaborative",
                    owner_username=(str(row.get("owner_username", "")).strip() or None),
                    dataset_token=(str(row.get("dataset_token", "")).strip() or None),
                    active=bool(row.get("active", True)),
                )
            )
        except Exception:
            continue
    return projects


def _load_user_access_from_file(user_access_file_path: str | None) -> dict[str, dict[str, Role]]:
    if not user_access_file_path:
        return {}

    path = Path(user_access_file_path)
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    result: dict[str, dict[str, Role]] = {}
    for username, roles_payload in payload.items():
        if not isinstance(roles_payload, dict):
            continue
        normalized_roles: dict[str, Role] = {}
        for project_slug, role_value in roles_payload.items():
            role_text = str(role_value).strip().lower()
            if role_text not in {"admin", "validator"}:
                continue
            normalized_roles[str(project_slug)] = Role(role_text)
        result[str(username)] = normalized_roles
    return result


def _load_pending_invites_from_file(invites_file_path: str | None) -> dict[str, dict[str, dict[str, str]]]:
    if not invites_file_path:
        return {}

    path = Path(invites_file_path)
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def _resolve_bootstrap_file_paths(runtime_config: RuntimeConfig) -> tuple[Path, Path, Path]:
    bootstrap_dir = Path(runtime_config.bootstrap_base_dir)
    projects_path = Path(runtime_config.projects_file_path) if runtime_config.projects_file_path else (bootstrap_dir / "projects.json")
    user_access_path = Path(runtime_config.user_access_file_path) if runtime_config.user_access_file_path else (bootstrap_dir / "user_access.json")
    invites_path = Path(runtime_config.invites_file_path) if runtime_config.invites_file_path else (bootstrap_dir / "invites.json")
    return projects_path, user_access_path, invites_path


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _persist_bootstrap_state(
    projects_path: Path,
    user_access_path: Path,
    invites_path: Path,
    admin_manager: AdminPanelManager,
    auth_service: AuthService,
    state_store: SupabaseBootstrapStore | None = None,
) -> None:
    project_rows = admin_manager.list_projects()
    projects_payload = [
        {
            "project_id": str(project.get("project_id", "")).strip(),
            "project_slug": str(project.get("project_slug", "")).strip(),
            "name": str(project.get("name", "")).strip(),
            "dataset_repo_id": str(project.get("dataset_repo_id", "")).strip(),
            "visibility": str(project.get("visibility", "collaborative")).strip() or "collaborative",
            "owner_username": str(project.get("owner_username", "")).strip() or None,
            "dataset_token": str(project.get("dataset_token", "")).strip() or None,
            "active": bool(project.get("active", True)),
        }
        for project in project_rows
    ]
    access_payload = auth_service.export_user_access_map(include_inactive=True)
    invites_payload = auth_service.export_pending_invites_map()

    if state_store is not None:
        state_store.persist(projects_payload, access_payload, invites_payload)
        return

    _atomic_write_json(projects_path, projects_payload)
    _atomic_write_json(user_access_path, access_payload)
    _atomic_write_json(invites_path, invites_payload)


def _bootstrap_auth_and_projects(
    auth_service: AuthService,
    admin_manager: AdminPanelManager,
    runtime_config: RuntimeConfig,
    projects_file_path: str | None = None,
    user_access_file_path: str | None = None,
    invites_file_path: str | None = None,
    state_store: SupabaseBootstrapStore | None = None,
) -> str:
    if state_store is not None:
        projects = state_store.load_projects()
        user_access = state_store.load_user_access()
        pending_invites = state_store.load_pending_invites()
    else:
        projects = _load_projects_from_file(projects_file_path or runtime_config.projects_file_path)
        user_access = _load_user_access_from_file(user_access_file_path or runtime_config.user_access_file_path)
        pending_invites = _load_pending_invites_from_file(invites_file_path or runtime_config.invites_file_path)

    if runtime_config.enable_demo_bootstrap and not projects and not user_access:
        projects = _default_projects()
        user_access = _default_user_access()

    for project in projects:
        _ = admin_manager.register_project(project)

    for username, access in user_access.items():
        auth_service.register_user_project_access(username, access)

    # Enforce private-project owner-only ACL even if bootstrap files are malformed.
    for username in auth_service.list_usernames(include_inactive=True):
        for project_slug in list(auth_service.list_user_projects(username)):
            project = admin_manager.get_project(project_slug)
            if project is None:
                continue
            if project.visibility != "private":
                continue
            owner = (project.owner_username or "").strip()
            if not owner or username != owner:
                auth_service.remove_user_project_role(username, project_slug)

    auth_service.load_pending_invites_map(pending_invites)
    for invite in auth_service.list_all_pending_invites():
        project = admin_manager.get_project(invite.project_slug)
        if project is None:
            continue
        if project.visibility == "private":
            auth_service.revoke_project_invite(invite.username, invite.project_slug)

    emergency_admin_message = ""
    has_admin = any(
        role == "admin"
        for roles in auth_service.export_user_access_map(include_inactive=True).values()
        for role in roles.values()
    )
    if projects and not has_admin:
        emergency_admin_username = "admin_user"
        for project in projects:
            auth_service.upsert_user_project_role(emergency_admin_username, project.project_slug, Role.admin)
        auth_service.set_user_active(emergency_admin_username, True)
        emergency_admin_message = (
            "⚠️ No administrator was configured in bootstrap files. "
            "Emergency admin access was granted to username 'admin_user'."
        )

    if not projects:
        return ""

    return emergency_admin_message


def _build_supabase_state(runtime_config: RuntimeConfig) -> tuple[SupabaseBootstrapStore | None, SupabaseValidationRepository | None, str]:
    backend = (runtime_config.state_backend or "filesystem").strip().lower()
    if backend not in {"supabase", "postgres", "postgresql"}:
        return None, None, ""

    if not runtime_config.supabase_url or not runtime_config.supabase_service_role_key:
        return (
            None,
            None,
            "⚠️ BIRDNET_STATE_BACKEND=supabase, but SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY is missing. Falling back to local files.",
        )

    try:
        client = SupabaseRestClient(
            url=runtime_config.supabase_url,
            service_role_key=runtime_config.supabase_service_role_key,
        )
        bootstrap_store = SupabaseBootstrapStore(client)
        validation_repository = SupabaseValidationRepository(client)
        return bootstrap_store, validation_repository, "Supabase state backend enabled."
    except Exception as exc:
        return None, None, f"⚠️ Could not initialize Supabase state backend: {exc}. Falling back to local files."


def _page_to_table(
    service: _QueueServiceProtocol,
    snapshot_reader: _ValidationReadRepositoryProtocol,
    project_slug: str,
    page: int,
    scientific_name: str,
    min_confidence: float,
    page_size: int = 10,
    validator_filter: str = "",
    status_filter: str = "all",
    updated_after: object = None,
    conflict_detection_key: str = "",
    show_conflicts_only: bool = False,
):
    filter_name = scientific_name.strip() if scientific_name.strip() else None
    snapshot = snapshot_reader.load_current_snapshot(project_slug=project_slug)
    normalized_status_filter = status_filter.strip().lower() if status_filter else "all"
    normalized_validator_filter = validator_filter.strip().lower()

    list_all = getattr(service, "list_all_detections", None)
    if callable(list_all):
        items = list_all(
            project_slug=project_slug,
            scientific_name=filter_name,
            min_confidence=min_confidence,
            max_confidence=None,
        )
    else:
        # Compatibility path for light test doubles. Real services expose
        # list_all_detections so status/validator/date filters can be applied
        # before pagination.
        first_page = service.get_page(
            project_slug=project_slug,
            page=1,
            page_size=page_size,
            scientific_name=filter_name,
            min_confidence=min_confidence,
        )
        items = list(getattr(first_page, "items", []))
        next_page = 2
        while bool(getattr(first_page, "has_next", False)):
            first_page = service.get_page(
                project_slug=project_slug,
                page=next_page,
                page_size=page_size,
                scientific_name=filter_name,
                min_confidence=min_confidence,
            )
            items.extend(list(getattr(first_page, "items", [])))
            next_page += 1

    updated_after_date: date | None = None
    if updated_after is not None:
        if isinstance(updated_after, datetime):
            updated_after_date = updated_after.date()
        elif isinstance(updated_after, date):
            updated_after_date = updated_after
        elif isinstance(updated_after, (int, float)):
            updated_after_date = datetime.fromtimestamp(float(updated_after)).date()
        else:
            updated_after_text = str(updated_after).strip()
            if updated_after_text:
                try:
                    updated_after_date = datetime.strptime(updated_after_text, "%Y-%m-%d").date()
                except ValueError:
                    try:
                        updated_after_date = datetime.fromisoformat(updated_after_text.replace("Z", "+00:00")).date()
                    except ValueError:
                        updated_after_date = None

    rows = [
        [
            item.detection_key,
            item.audio_id,
            item.scientific_name,
            round(item.confidence, 3),
            item.start_time,
            item.end_time,
            str(snapshot.get(item.detection_key, {}).get("status", "pending")),
            int(snapshot.get(item.detection_key, {}).get("version", 0)),
            "CONFLICT" if conflict_detection_key and item.detection_key == conflict_detection_key else "",
            "HIGH" if conflict_detection_key and item.detection_key == conflict_detection_key else "",
        ]
        for item in items
    ]

    if normalized_validator_filter:
        rows = [
            row
            for row in rows
            if normalized_validator_filter in str(snapshot.get(str(row[0]), {}).get("validator", "")).strip().lower()
        ]

    if normalized_status_filter and normalized_status_filter != "all":
        rows = [row for row in rows if str(row[6]).strip().lower() == normalized_status_filter]

    if updated_after_date:
        filtered_rows: list[list[object]] = []
        for row in rows:
            snapshot_item = snapshot.get(str(row[0]), {})
            updated_at_value = str(snapshot_item.get("updated_at", "")).strip()
            if not updated_at_value:
                continue
            try:
                item_date = datetime.fromisoformat(updated_at_value.replace("Z", "+00:00")).date()
                if item_date >= updated_after_date:
                    filtered_rows.append(row)
            except ValueError:
                continue
        rows = filtered_rows

    if show_conflicts_only:
        rows = [row for row in rows if str(row[8]) == "CONFLICT"]

    rows = _sort_rows_by_confidence_desc(rows)
    filtered_total = len(rows)
    total_pages = max(1, ((filtered_total - 1) // page_size) + 1) if filtered_total else 1
    safe_page = max(1, min(int(page), total_pages))
    start = (safe_page - 1) * page_size
    page_rows = rows[start : start + page_size]

    status = (
        f"Page {safe_page}/{total_pages} | Base total: {len(items)} | "
        f"Filtered: {filtered_total} | Shown: {len(page_rows)}"
    )
    if show_conflicts_only:
        status = f"{status} | Conflicts only: {filtered_total} item(ns)"
    return page_rows, status, safe_page


def _get_project_detection_count(service: _QueueServiceProtocol, project_slug: str) -> int:
    if not project_slug:
        return 0

    try:
        page_obj = service.get_page(
            project_slug=project_slug,
            page=1,
            page_size=1,
        )
        return int(getattr(page_obj, "total_items", 0))
    except Exception:
        return 0


def _build_queue_badge(service: _QueueServiceProtocol, project_slug: str | None) -> str:
    if not project_slug:
        return "<span class='bn-pill'>Queue: --</span>"

    total = _get_project_detection_count(service, project_slug)
    return (
        "<span class='bn-pill bn-pill-ok'>"
        f"Queue: {total}"
        "</span>"
    )


def _build_validation_report(snapshot_reader: _ValidationReadRepositoryProtocol, project_slug: str) -> str:
    snapshot = snapshot_reader.load_current_snapshot(project_slug=project_slug)
    events = snapshot_reader.list_events(project_slug=project_slug)

    counts: dict[str, int] = {}
    for payload in snapshot.values():
        status_value = str(payload.get("status", "unknown"))
        counts[status_value] = counts.get(status_value, 0) + 1

    parts = [
        f"Project: {project_slug}",
        f"Append-only events: {len(events)}",
        f"Detections with current state: {len(snapshot)}",
    ]
    if counts:
        summary = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        parts.append(f"Current status: {summary}")
    else:
        parts.append("Current status: no validations")
    return " | ".join(parts)


def _extract_audio_id(rows: object, selected_index: int) -> str:
    normalized_rows: list[list[object]]

    if hasattr(rows, "values"):
        normalized_rows = [list(item) for item in rows.values.tolist()]
    else:
        normalized_rows = [list(item) for item in rows] if rows else []

    if not normalized_rows:
        raise ValueError("No detections loaded in table")
    if selected_index < 0 or selected_index >= len(normalized_rows):
        raise ValueError("Select a valid detection row in table")

    value = normalized_rows[selected_index][1]
    audio_id = str(value).strip()
    if not audio_id:
        raise ValueError("Invalid audio_id in selected detection")
    return audio_id


def _extract_detection_key(rows: object, selected_index: int) -> str:
    normalized_rows: list[list[object]]

    if hasattr(rows, "values"):
        normalized_rows = [list(item) for item in rows.values.tolist()]
    else:
        normalized_rows = [list(item) for item in rows] if rows else []

    if not normalized_rows:
        raise ValueError("No detections loaded in table")
    if selected_index < 0 or selected_index >= len(normalized_rows):
        raise ValueError("Select a valid detection row in table")

    value = normalized_rows[selected_index][0]
    detection_key = str(value).strip()
    if not detection_key:
        raise ValueError("Invalid detection_key in selected detection")
    return detection_key


def _find_detection_row_index(rows: object, detection_key: str) -> int:
    normalized_rows: list[list[object]]

    if hasattr(rows, "values"):
        normalized_rows = [list(item) for item in rows.values.tolist()]
    else:
        normalized_rows = [list(item) for item in rows] if rows else []

    for index, row in enumerate(normalized_rows):
        if str(row[0]).strip() == detection_key:
            return index
    return 0


def _post_validation_queue_anchor(rows: object, detection_key: str, previous_index: int) -> int:
    normalized_rows = _normalize_rows(rows)
    if not normalized_rows:
        return 0

    for index, row in enumerate(normalized_rows):
        if str(row[0]).strip() == detection_key:
            return index

    # The saved row can disappear under pending/status filters after refresh.
    # Anchor one row behind its former position so the follow-up advance opens
    # the row that slid into the saved row's place.
    return min(int(previous_index), len(normalized_rows)) - 1


def _extract_expected_version(rows: object, selected_index: int) -> int:
    normalized_rows: list[list[object]]

    if hasattr(rows, "values"):
        normalized_rows = [list(item) for item in rows.values.tolist()]
    else:
        normalized_rows = [list(item) for item in rows] if rows else []

    if not normalized_rows:
        raise ValueError("No detections loaded in table")
    if selected_index < 0 or selected_index >= len(normalized_rows):
        raise ValueError("Select a valid detection row in table")

    value = normalized_rows[selected_index][7]
    return int(value)


def _fetch_selected_audio(
    audio_service: _AudioServiceProtocol,
    dataset_repo: str,
    rows: object,
    selected_index: int,
    previous_cache_key: str,
    allow_demo_fallback: bool = False,
    hf_token: str | None = None,
) -> tuple[str | None, str, str]:
    repo = dataset_repo.strip()
    if not repo:
        return None, "", "Provide dataset repo in owner/repo format. Example: org/dataset-name"

    try:
        audio_id = _extract_audio_id(rows=rows, selected_index=selected_index)
        try:
            result = audio_service.fetch(
                dataset_repo=repo,
                audio_id=audio_id,
                allow_demo_fallback=allow_demo_fallback,
                hf_token=hf_token,
            )
        except TypeError:
            # Backward compatibility for fake/mocked services used in tests.
            result = audio_service.fetch(dataset_repo=repo, audio_id=audio_id)
        status = f"Audio loaded ({result.source}) for audio_id={audio_id}"
        return result.local_path, result.cache_key, status
    except Exception as exc:
        if previous_cache_key:
            return None, previous_cache_key, f"Failed to load audio: {exc}"
        return None, "", f"Failed to load audio: {exc}"


def _load_pcm_wave(audio_path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(audio_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported PCM width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return sample_rate, audio


def _magma_like_colormap(normalized: np.ndarray) -> np.ndarray:
    # Lightweight gradient for spectrogram rendering (dark -> orange -> yellow).
    anchors = np.array(
        [
            [0, 0, 4],
            [28, 16, 68],
            [79, 18, 123],
            [129, 37, 129],
            [181, 54, 122],
            [229, 80, 100],
            [251, 135, 97],
            [254, 194, 135],
            [252, 253, 191],
        ],
        dtype=np.float32,
    )
    indices = np.clip((normalized * (len(anchors) - 1)).astype(np.float32), 0, len(anchors) - 1)
    lower = np.floor(indices).astype(np.int32)
    upper = np.clip(lower + 1, 0, len(anchors) - 1)
    blend = (indices - lower)[..., np.newaxis]
    rgb = anchors[lower] * (1.0 - blend) + anchors[upper] * blend
    return rgb.astype(np.uint8)


def _build_spectrogram_image(audio_path: str | None) -> str | None:
    if not audio_path:
        return None

    source_path = Path(audio_path)
    if not source_path.exists() or source_path.suffix.lower() != ".wav":
        return None

    try:
        _, samples = _load_pcm_wave(source_path)
    except Exception:
        return None

    if samples.size < 1024:
        return None

    window_size = 512
    hop_size = 128
    frame_count = 1 + max(0, (len(samples) - window_size) // hop_size)
    if frame_count <= 0:
        return None

    frames = np.lib.stride_tricks.sliding_window_view(samples, window_shape=window_size)[::hop_size]
    if frames.size == 0:
        return None

    window = np.hanning(window_size).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1))
    spectrum = np.maximum(spectrum, 1e-9)
    db = 20.0 * np.log10(spectrum)
    db = db.T
    low = float(np.percentile(db, 5))
    high = float(np.percentile(db, 99))
    if high <= low:
        return None

    normalized = np.clip((db - low) / (high - low), 0.0, 1.0)
    rgb = _magma_like_colormap(normalized)
    rgb = np.flipud(rgb)

    image = Image.fromarray(rgb, mode="RGB").resize((900, 320), resample=Image.Resampling.BICUBIC)
    cache_name = hashlib.sha1(f"{audio_path}:{source_path.stat().st_mtime_ns}".encode("utf-8")).hexdigest()[:16]
    output_path = Path(tempfile.gettempdir()) / f"birdnet_validator_spec_{cache_name}.png"
    image.save(output_path)
    return str(output_path)


def _fetch_selected_audio_with_spectrogram(
    audio_service: _AudioServiceProtocol,
    dataset_repo: str,
    rows: object,
    selected_index: int,
    previous_cache_key: str,
    allow_demo_fallback: bool = False,
    hf_token: str | None = None,
) -> tuple[str | None, str, str, str | None]:
    audio_path, cache_key, status = _fetch_selected_audio(
        audio_service=audio_service,
        dataset_repo=dataset_repo,
        rows=rows,
        selected_index=selected_index,
        previous_cache_key=previous_cache_key,
        allow_demo_fallback=allow_demo_fallback,
        hf_token=hf_token,
    )
    spectrogram_path = _build_spectrogram_image(audio_path)
    if audio_path and spectrogram_path is None:
        status = f"{status} | Spectrogram unavailable (requires WAV)."
    return audio_path, cache_key, status, spectrogram_path


def _build_validation_summary_cards(rows: object) -> str:
    if hasattr(rows, "values"):
        normalized_rows = [list(item) for item in rows.values.tolist()]
    else:
        normalized_rows = [list(item) for item in rows] if rows else []
    total = len(normalized_rows)
    positive = 0
    negative = 0
    uncertain = 0
    skipped = 0
    for row in normalized_rows:
        status_value = str(row[6]).strip().lower() if len(row) > 6 else ""
        if status_value == "positive":
            positive += 1
        elif status_value == "negative":
            negative += 1
        elif status_value == "uncertain":
            uncertain += 1
        elif status_value == "skip":
            skipped += 1

    reviewed = positive + negative + uncertain + skipped
    pending = max(0, total - reviewed)
    reviewed_pct = round((reviewed / total) * 100, 1) if total else 0.0

    return (
        compact_metric_grid(
            [
                ("Queue page", str(total), "loaded items", "info"),
                ("Accepted", str(positive), "positive validations", "positive"),
                ("Rejected", str(negative), "negative validations", "negative"),
                ("Reviewed", f"{reviewed_pct}%", f"{pending} pending on page", "warning"),
            ]
        )
    )


def _autofetch_first_row(
    audio_service: _AudioServiceProtocol,
    dataset_repo: str,
    rows: object,
    cache_key: str,
    allow_demo_fallback: bool = False,
    hf_token: str | None = None,
) -> tuple[int, str | None, str, str, str | None]:
    normalized_rows = _normalize_rows(rows)
    if not normalized_rows:
        return 0, None, "", "No detections available to auto-load audio", None

    audio_path, updated_cache_key, status, spectrogram_path = _fetch_selected_audio_with_spectrogram(
        audio_service=audio_service,
        dataset_repo=dataset_repo,
        rows=normalized_rows,
        selected_index=0,
        previous_cache_key=cache_key,
        allow_demo_fallback=allow_demo_fallback,
        hf_token=hf_token,
    )
    return 0, audio_path, updated_cache_key, status, spectrogram_path


def _select_and_fetch_audio(
    audio_service: _AudioServiceProtocol,
    dataset_repo: str,
    rows: object,
    cache_key: str,
    evt: gr.SelectData,
    allow_demo_fallback: bool = False,
    hf_token: str | None = None,
) -> tuple[int, str | None, str, str, str | None]:
    selected_index = _selected_dataframe_row_index(rows, evt)

    audio_path, updated_cache_key, status, spectrogram_path = _fetch_selected_audio_with_spectrogram(
        audio_service=audio_service,
        dataset_repo=dataset_repo,
        rows=rows,
        selected_index=selected_index,
        previous_cache_key=cache_key,
        allow_demo_fallback=allow_demo_fallback,
        hf_token=hf_token,
    )
    return selected_index, audio_path, updated_cache_key, status, spectrogram_path


def _selected_dataframe_row_index(rows: object, evt: gr.SelectData) -> int:
    normalized_rows = _normalize_rows(rows)
    if not normalized_rows:
        return 0

    row_value = getattr(evt, "row_value", None)
    if isinstance(row_value, (list, tuple)) and row_value:
        detection_key = str(row_value[0]).strip()
        if detection_key:
            return _find_detection_row_index(normalized_rows, detection_key)

    raw_index = getattr(evt, "index", None)
    if isinstance(raw_index, int):
        return max(0, min(raw_index, len(normalized_rows) - 1))
    if isinstance(raw_index, tuple):
        numeric_candidates = [int(value) for value in raw_index if isinstance(value, int)]
        # Gradio Dataframe events carry a 2-D cell index. If row_value is not
        # available, prefer the only coordinate that is a valid table row.
        valid_candidates = [value for value in numeric_candidates if 0 <= value < len(normalized_rows)]
        if valid_candidates:
            return valid_candidates[-1]
    return 0


def _normalize_rows(rows: object) -> list[list[object]]:
    if hasattr(rows, "values"):
        return [list(item) for item in rows.values.tolist()]
    return [list(item) for item in rows] if rows else []


def _spectrogram_title(species_name: str | None, confidence: float | None) -> str:
    _ = species_name
    _ = confidence
    return "### Segment spectrogram"


def _selected_row_species_and_confidence(rows: object, selected_index: int) -> tuple[str | None, float | None]:
    normalized_rows = _normalize_rows(rows)
    if normalized_rows:
        safe_index = max(0, min(int(selected_index), len(normalized_rows) - 1))
        row = normalized_rows[safe_index]

        species_name: str | None = None
        confidence_value: float | None = None

        if len(row) > 2:
            raw_species = str(row[2]).strip()
            if raw_species.startswith("▶ "):
                raw_species = raw_species[2:].strip()
            species_name = raw_species or None

        if len(row) > 3:
            try:
                confidence_value = float(row[3])
            except Exception:
                confidence_value = None

        return species_name, confidence_value

    return None, None


def _mark_selected_row(rows: object, selected_index: int) -> list[list[object]]:
    normalized_rows = _normalize_rows(rows)
    if not normalized_rows:
        return []

    safe_index = max(0, min(int(selected_index), len(normalized_rows) - 1))
    marked_rows: list[list[object]] = []
    for row_index, row in enumerate(normalized_rows):
        updated = list(row)
        if len(updated) > 2:
            species = str(updated[2])
            if species.startswith("▶ "):
                species = species[2:]
            if row_index == safe_index:
                species = f"▶ {species}"
            updated[2] = species
        marked_rows.append(updated)

    return marked_rows


def _selected_segment_card(rows: object, selected_index: int) -> str:
    normalized_rows = _normalize_rows(rows)
    if not normalized_rows:
        return selected_segment_html(None)
    safe_index = max(0, min(int(selected_index), len(normalized_rows) - 1))
    return selected_segment_html(normalized_rows[safe_index], safe_index, len(normalized_rows))


def _extract_species_options_from_queue(
    queue_service: _QueueServiceProtocol,
    project_slug: str,
    page_size: int,
) -> list[str]:
    if not project_slug:
        return []

    species_set: set[str] = set()
    page = 1
    while True:
        page_result = queue_service.get_page(
            project_slug=project_slug,
            page=page,
            page_size=page_size,
            scientific_name=None,
            min_confidence=None,
            max_confidence=None,
        )
        for item in page_result.items:
            name = str(item.scientific_name).strip()
            if name:
                species_set.add(name)
        if not page_result.has_next:
            break
        page += 1

    return sorted(species_set)


def _sort_rows_by_confidence_desc(rows: list[list[object]]) -> list[list[object]]:
    return sorted(rows, key=lambda row: float(row[3]) if len(row) > 3 else 0.0, reverse=True)


def _select_and_fetch_audio_with_title(
    audio_service: _AudioServiceProtocol,
    dataset_repo: str,
    rows: object,
    cache_key: str,
    evt: gr.SelectData,
    allow_demo_fallback: bool = False,
    hf_token: str | None = None,
) -> tuple[int, str | None, str, str, str | None, str]:
    selected_index, audio_path, updated_cache_key, status, spectrogram_path = _select_and_fetch_audio(
        audio_service=audio_service,
        dataset_repo=dataset_repo,
        rows=rows,
        cache_key=cache_key,
        evt=evt,
        allow_demo_fallback=allow_demo_fallback,
        hf_token=hf_token,
    )
    species_name, confidence_value = _selected_row_species_and_confidence(rows, selected_index)
    return (
        selected_index,
        audio_path,
        updated_cache_key,
        status,
        spectrogram_path,
        _spectrogram_title(species_name, confidence_value),
    )


def _autofetch_first_row_with_title(
    audio_service: _AudioServiceProtocol,
    dataset_repo: str,
    rows: object,
    cache_key: str,
    allow_demo_fallback: bool = False,
    hf_token: str | None = None,
) -> tuple[int, str | None, str, str, str | None, str]:
    selected_index, audio_path, updated_cache_key, status, spectrogram_path = _autofetch_first_row(
        audio_service=audio_service,
        dataset_repo=dataset_repo,
        rows=rows,
        cache_key=cache_key,
        allow_demo_fallback=allow_demo_fallback,
        hf_token=hf_token,
    )
    species_name, confidence_value = _selected_row_species_and_confidence(rows, selected_index)
    return (
        selected_index,
        audio_path,
        updated_cache_key,
        status,
        spectrogram_path,
        _spectrogram_title(species_name, confidence_value),
    )


def _advance_to_next_row_with_title(
    audio_service: _AudioServiceProtocol,
    dataset_repo: str,
    rows: object,
    selected_index: int,
    cache_key: str,
    allow_demo_fallback: bool = False,
    hf_token: str | None = None,
) -> tuple[int, str | None, str, str, str | None, str]:
    normalized_rows = _normalize_rows(rows)
    if not normalized_rows:
        return 0, None, cache_key, "No detections available", None, _spectrogram_title(None, None)

    safe_index = int(selected_index) + 1
    if safe_index >= len(normalized_rows):
        safe_index = _first_pending_row_index(normalized_rows)
    safe_index = max(0, safe_index)
    audio_path, updated_cache_key, status, spectrogram_path = _fetch_selected_audio_with_spectrogram(
        audio_service=audio_service,
        dataset_repo=dataset_repo,
        rows=normalized_rows,
        selected_index=safe_index,
        previous_cache_key=cache_key,
        allow_demo_fallback=allow_demo_fallback,
        hf_token=hf_token,
    )
    species_name, confidence_value = _selected_row_species_and_confidence(normalized_rows, safe_index)
    return safe_index, audio_path, updated_cache_key, status, spectrogram_path, _spectrogram_title(species_name, confidence_value)


def _first_pending_row_index(rows: object) -> int:
    normalized_rows = _normalize_rows(rows)
    for index, row in enumerate(normalized_rows):
        status_value = str(row[6] if len(row) > 6 else "pending").strip().lower()
        if status_value in {"", "pending"}:
            return index
    return 0


def _first_pending_queue_page(
    queue_service: _QueueServiceProtocol,
    snapshot_reader: _ValidationReadRepositoryProtocol,
    project_slug: str,
    scientific_name: str,
    min_confidence: float,
    validator_filter: str,
    status_filter: str,
    updated_after: object,
    show_conflicts_only: bool,
) -> tuple[list[list[object]], int, int] | None:
    normalized_status_filter = (status_filter or "all").strip().lower()
    if normalized_status_filter not in {"", "all", "pending"}:
        return None

    requested_page = 1
    visited_pages: set[int] = set()
    while True:
        page_rows, _, actual_page = _page_to_table(
            service=queue_service,
            snapshot_reader=snapshot_reader,
            project_slug=project_slug,
            page=requested_page,
            scientific_name=scientific_name,
            min_confidence=min_confidence,
            validator_filter=validator_filter,
            status_filter=status_filter,
            updated_after=updated_after,
            show_conflicts_only=show_conflicts_only,
        )
        if actual_page in visited_pages:
            return None
        visited_pages.add(actual_page)

        page_rows = _sort_rows_by_confidence_desc(page_rows)
        pending_index = _first_pending_row_index(page_rows)
        if page_rows:
            status_value = str(page_rows[pending_index][6] if len(page_rows[pending_index]) > 6 else "pending").strip().lower()
            if status_value in {"", "pending"}:
                return page_rows, actual_page, pending_index

        requested_page = actual_page + 1


def _cleanup_selected_audio(audio_service: _AudioServiceProtocol, cache_key: str) -> tuple[str, str | None]:
    if not cache_key:
        return "No cached audio to clean", None

    audio_service.cleanup_after_validation(cache_key=cache_key)
    return "Audio cache cleaned after validation", None


def _validator_name_from_session(session: object) -> str:
    return str(getattr(session, "username", "") or "").strip()


def _save_selected_validation(
    validation_service: _ValidationServiceProtocol,
    audio_service: _AudioServiceProtocol,
    project_slug: str,
    rows: object,
    selected_index: int,
    status_value: str,
    validator: str,
    notes: str,
    cache_key: str,
    corrected_species: str | None = None,
) -> tuple[str, str, str | None]:
    validator_name = validator.strip()
    if not validator_name:
        return "Provide validator name before saving", cache_key, None

    try:
        detection_key = _extract_detection_key(rows=rows, selected_index=selected_index)
        expected_version = _extract_expected_version(rows=rows, selected_index=selected_index)
        _ = validation_service.validate_detection(
            project_slug=project_slug,
            detection_key=detection_key,
            status=status_value,
            validator=validator_name,
            notes=notes.strip(),
            corrected_species=(corrected_species or "").strip() or None,
            expected_version=expected_version,
        )
        if cache_key:
            audio_service.cleanup_after_validation(cache_key=cache_key)
        return f"Validation saved: {detection_key} -> {status_value}", "", None
    except OptimisticLockError as exc:
        return (
            "Concurrency conflict: this detection was updated by another validator "
            f"(detection_key={exc.detection_key}, current version={exc.current_version}, expected={exc.expected_version}). "
            "Refresh the table.",
            cache_key,
            None,
        )
    except Exception as exc:
        return f"Failed to save validation: {exc}", cache_key, None


def _save_selected_validation_with_refresh(
    validation_service: _ValidationServiceProtocol,
    audio_service: _AudioServiceProtocol,
    queue_service: _QueueServiceProtocol,
    snapshot_reader: _ValidationReadRepositoryProtocol,
    project_slug: str,
    rows: object,
    selected_index: int,
    status_value: str,
    validator: str,
    notes: str,
    cache_key: str,
    page: int,
    scientific_name: str,
    min_confidence: float,
    validator_filter: str,
    status_filter: str,
    updated_after: object,
    show_conflicts_only: bool,
    corrected_species: str | None = None,
) -> tuple[str, str, str | None, list[list[object]], int, int, str, str]:
    prior_rows = _normalize_rows(rows)
    selected_was_last_row = bool(prior_rows) and int(selected_index) >= len(prior_rows) - 1
    selected_key = ""
    try:
        selected_key = _extract_detection_key(rows=rows, selected_index=selected_index)
    except Exception:
        selected_key = ""

    save_status, updated_cache_key, audio_path = _save_selected_validation(
        validation_service=validation_service,
        audio_service=audio_service,
        project_slug=project_slug,
        rows=rows,
        selected_index=selected_index,
        status_value=status_value,
        validator=validator,
        notes=notes,
        cache_key=cache_key,
        corrected_species=corrected_species,
    )

    refreshed_rows, page_status, refreshed_page = _page_to_table(
        service=queue_service,
        snapshot_reader=snapshot_reader,
        project_slug=project_slug,
        page=page,
        scientific_name=scientific_name,
        min_confidence=min_confidence,
        validator_filter=validator_filter,
        status_filter=status_filter,
        updated_after=updated_after,
        show_conflicts_only=show_conflicts_only,
    )
    refreshed_rows = _sort_rows_by_confidence_desc(refreshed_rows)

    if selected_key:
        refreshed_index = _post_validation_queue_anchor(refreshed_rows, selected_key, selected_index)
    else:
        refreshed_index = 0

    if "Concurrency conflict" in save_status:
        conflict_key = selected_key
        refreshed_rows, page_status, refreshed_page = _page_to_table(
            service=queue_service,
            snapshot_reader=snapshot_reader,
            project_slug=project_slug,
            page=refreshed_page,
            scientific_name=scientific_name,
            min_confidence=min_confidence,
            validator_filter=validator_filter,
            status_filter=status_filter,
            updated_after=updated_after,
            conflict_detection_key=conflict_key,
            show_conflicts_only=show_conflicts_only,
        )
        refreshed_rows = _sort_rows_by_confidence_desc(refreshed_rows)
        refreshed_index = _find_detection_row_index(refreshed_rows, selected_key) if selected_key else 0
        pending_status_value = status_value
        status = f"{save_status} Table reloaded to resolve conflict."
    else:
        conflict_key = ""
        pending_status_value = ""
        if selected_was_last_row:
            first_pending_page = _first_pending_queue_page(
                queue_service=queue_service,
                snapshot_reader=snapshot_reader,
                project_slug=project_slug,
                scientific_name=scientific_name,
                min_confidence=min_confidence,
                validator_filter=validator_filter,
                status_filter=status_filter,
                updated_after=updated_after,
                show_conflicts_only=show_conflicts_only,
            )
            if first_pending_page is not None:
                refreshed_rows, refreshed_page, pending_index = first_pending_page
                refreshed_index = pending_index - 1
        status = f"{save_status} | {page_status}"

    return (
        status,
        updated_cache_key,
        audio_path,
        refreshed_rows,
        refreshed_page,
        refreshed_index,
        pending_status_value,
        conflict_key,
    )


def _reapply_last_conflict_validation_with_refresh(
    validation_service: _ValidationServiceProtocol,
    audio_service: _AudioServiceProtocol,
    queue_service: _QueueServiceProtocol,
    snapshot_reader: _ValidationReadRepositoryProtocol,
    project_slug: str,
    rows: object,
    selected_index: int,
    pending_status_value: str,
    conflict_detection_key: str,
    validator: str,
    notes: str,
    cache_key: str,
    page: int,
    scientific_name: str,
    min_confidence: float,
    validator_filter: str,
    status_filter: str,
    updated_after: object,
    show_conflicts_only: bool,
) -> tuple[str, str, str | None, list[list[object]], int, int, str, str]:
    if not pending_status_value:
        refreshed_rows, page_status, refreshed_page = _page_to_table(
            service=queue_service,
            snapshot_reader=snapshot_reader,
            project_slug=project_slug,
            page=page,
            scientific_name=scientific_name,
            min_confidence=min_confidence,
            validator_filter=validator_filter,
            status_filter=status_filter,
            updated_after=updated_after,
            show_conflicts_only=show_conflicts_only,
        )
        refreshed_rows = _sort_rows_by_confidence_desc(refreshed_rows)
        return (
            f"No pending validation to reapply | {page_status}",
            cache_key,
            None,
            refreshed_rows,
            refreshed_page,
            selected_index,
            "",
            "",
        )

    target_index = _find_detection_row_index(rows, conflict_detection_key) if conflict_detection_key else selected_index
    return _save_selected_validation_with_refresh(
        validation_service=validation_service,
        audio_service=audio_service,
        queue_service=queue_service,
        snapshot_reader=snapshot_reader,
        project_slug=project_slug,
        rows=rows,
        selected_index=target_index,
        status_value=pending_status_value,
        validator=validator,
        notes=notes,
        cache_key=cache_key,
        page=page,
        scientific_name=scientific_name,
        min_confidence=min_confidence,
        validator_filter=validator_filter,
        status_filter=status_filter,
        updated_after=updated_after,
        show_conflicts_only=show_conflicts_only,
    )


def _batch_validate_conflicts(
    validation_service: _ValidationServiceProtocol,
    audio_service: _AudioServiceProtocol,
    queue_service: _QueueServiceProtocol,
    snapshot_reader: _ValidationReadRepositoryProtocol,
    project_slug: str,
    rows: object,
    status_value: str,
    validator: str,
    notes: str,
    cache_key: str,
    page: int,
    scientific_name: str,
    min_confidence: float,
    validator_filter: str,
    status_filter: str,
    updated_after: object,
) -> tuple[str, str, str | None, list[list[object]], int]:
    """Apply the same validation status to all visible conflicts in the table."""
    validator_name = validator.strip()
    if not validator_name:
        return "Provide validator name", "", None, [], page

    normalized_rows: list[list[object]]
    if hasattr(rows, "values"):
        normalized_rows = [list(item) for item in rows.values.tolist()]
    else:
        normalized_rows = [list(item) for item in rows] if rows else []

    if not normalized_rows:
        return "No conflict detection to validate", "", None, [], page

    conflict_rows = [row for row in normalized_rows if str(row[8]) == "CONFLICT"]
    if not conflict_rows:
        return "No conflict detection identified in table", "", None, normalized_rows, page

    success_count = 0
    failure_count = 0
    conflict_count = 0

    for row in conflict_rows:
        try:
            detection_key = str(row[0]).strip()
            expected_version = int(row[7])

            _ = validation_service.validate_detection(
                project_slug=project_slug,
                detection_key=detection_key,
                status=status_value,
                validator=validator_name,
                notes=notes.strip(),
                expected_version=expected_version,
            )
            success_count += 1
            if cache_key:
                audio_service.cleanup_after_validation(cache_key=cache_key)
        except OptimisticLockError:
            conflict_count += 1
        except Exception:
            failure_count += 1

    refreshed_rows, page_status, refreshed_page = _page_to_table(
        service=queue_service,
        snapshot_reader=snapshot_reader,
        project_slug=project_slug,
        page=page,
        scientific_name=scientific_name,
        min_confidence=min_confidence,
        validator_filter=validator_filter,
        status_filter=status_filter,
        updated_after=updated_after,
        show_conflicts_only=False,
    )
    refreshed_rows = _sort_rows_by_confidence_desc(refreshed_rows)

    summary = f"Processed {len(conflict_rows)} conflicts: {success_count} success, {conflict_count} new conflicts, {failure_count} failures"
    status = f"{summary} | {page_status}"

    return status, "", None, refreshed_rows, refreshed_page


_VALIDATION_EXPORT_BASE_COLUMNS = [
    "project_slug",
    "dataset_repo_id",
    "detection_key",
    "audio_id",
    "detection_scientific_name",
    "detection_confidence",
    "detection_start_time",
    "detection_end_time",
]

_VALIDATION_EXPORT_STATE_COLUMNS = [
    "validation_status",
    "validation_corrected_species",
    "validation_effective_species",
    "validation_notes",
    "validation_validator",
    "validation_updated_at",
    "validation_version",
    "validation_reviewed",
]

_XLSX_EXPORT_MAX_DATA_ROWS = 1_048_575
_VALIDATION_EXPORT_MAX_AGE_SECONDS = 6 * 60 * 60


def _cleanup_old_validation_exports(max_age_seconds: int = _VALIDATION_EXPORT_MAX_AGE_SECONDS) -> None:
    temp_root = Path(tempfile.gettempdir())
    cutoff = time.time() - max_age_seconds
    for export_dir in temp_root.glob("birdnet-validation-export-*"):
        try:
            if export_dir.is_dir() and export_dir.stat().st_mtime < cutoff:
                shutil.rmtree(export_dir, ignore_errors=True)
        except OSError:
            continue


def _export_cell_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, Mapping):
        return json.dumps(dict(value), ensure_ascii=True, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=True)
    return value


def _source_export_column(raw_key: object) -> str:
    key = str(raw_key or "").strip()
    return f"source_{key}" if key else ""


def _build_validation_export_rows(
    detections: list[Detection],
    snapshot: dict[str, dict[str, object]],
    *,
    project_slug: str,
    dataset_repo_id: str,
) -> tuple[list[str], list[dict[str, object]]]:
    """Join complete project detections with their current validation state."""
    source_columns: list[str] = []
    for detection in detections:
        for raw_key in detection.source_metadata:
            column = _source_export_column(raw_key)
            if column and column not in source_columns:
                source_columns.append(column)

    columns = [*_VALIDATION_EXPORT_BASE_COLUMNS, *source_columns, *_VALIDATION_EXPORT_STATE_COLUMNS]
    rows: list[dict[str, object]] = []
    sorted_detections = sorted(
        detections,
        key=lambda item: (
            item.scientific_name.lower(),
            item.audio_id.lower(),
            item.start_time,
            item.end_time,
            item.detection_key,
        ),
    )
    for detection in sorted_detections:
        state = snapshot.get(detection.detection_key, {})
        validation_status = str(state.get("status") or "pending").strip() or "pending"
        corrected_species = str(state.get("corrected_species") or "").strip()
        row: dict[str, object] = {
            "project_slug": project_slug,
            "dataset_repo_id": dataset_repo_id,
            "detection_key": detection.detection_key,
            "audio_id": detection.audio_id,
            "detection_scientific_name": detection.scientific_name,
            "detection_confidence": detection.confidence,
            "detection_start_time": detection.start_time,
            "detection_end_time": detection.end_time,
            "validation_status": validation_status,
            "validation_corrected_species": corrected_species,
            "validation_effective_species": corrected_species or detection.scientific_name,
            "validation_notes": str(state.get("notes") or ""),
            "validation_validator": str(state.get("validator") or ""),
            "validation_updated_at": str(state.get("updated_at") or ""),
            "validation_version": int(state.get("version") or 0),
            "validation_reviewed": validation_status.lower() != "pending",
        }
        for raw_key, raw_value in detection.source_metadata.items():
            column = _source_export_column(raw_key)
            if column:
                row[column] = _export_cell_value(raw_value)
        rows.append({column: _export_cell_value(row.get(column, "")) for column in columns})
    return columns, rows


def _write_validation_export(
    detections: list[Detection],
    snapshot: dict[str, dict[str, object]],
    *,
    project_slug: str,
    dataset_repo_id: str,
    file_format: str,
) -> Path:
    columns, rows = _build_validation_export_rows(
        detections,
        snapshot,
        project_slug=project_slug,
        dataset_repo_id=dataset_repo_id,
    )
    normalized_format = file_format.lower().strip()
    if normalized_format not in {"csv", "xlsx"}:
        raise ValueError("Export format must be csv or xlsx")

    _cleanup_old_validation_exports()

    safe_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", project_slug).strip("-") or "project"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(tempfile.mkdtemp(prefix="birdnet-validation-export-"))
    output_path = output_dir / f"{safe_slug}-validation-data-{timestamp}.{normalized_format}"

    if normalized_format == "csv":
        with output_path.open("w", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return output_path

    import pandas as pd  # type: ignore[import-not-found]

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if not rows:
            pd.DataFrame(columns=columns).to_excel(
                writer,
                sheet_name="validation_data",
                index=False,
                freeze_panes=(1, 0),
            )
        for chunk_index, start in enumerate(range(0, len(rows), _XLSX_EXPORT_MAX_DATA_ROWS), start=1):
            sheet_name = "validation_data" if chunk_index == 1 else f"validation_data_{chunk_index}"
            pd.DataFrame(rows[start : start + _XLSX_EXPORT_MAX_DATA_ROWS], columns=columns).to_excel(
                writer,
                sheet_name=sheet_name,
                index=False,
                freeze_panes=(1, 0),
            )
    return output_path


def create_app() -> gr.Blocks:
    """Build the BirdNET Validator app with multi-project auth integration.
    
    Returns multi-tab interface with:
    - Login tab for user authentication
    - Project selection for authorized projects
    - Admin panel for project/user management (admin only)
    - Validation interface for selected project
    
    Returns:
        Gradio Blocks with full auth-integrated app
    """
    runtime_config = RuntimeConfig.from_env()

    # Initialize auth service
    auth_service = AuthService(
        session_ttl_minutes=120,
        invite_ttl_hours=runtime_config.invite_ttl_hours,
    )

    # Initialize EmailJS invite notifier (only transport)
    invite_notifier: InviteEmailNotifier = EmailJSInviteEmailNotifier(
        sender_email=runtime_config.invite_email_sender,
        service_id=runtime_config.emailjs_service_id or "",
        template_id=runtime_config.emailjs_template_id or "",
        public_key=runtime_config.emailjs_public_key or "",
        template_id_username_only=runtime_config.emailjs_template_id_username_only,
        template_id_email_only=runtime_config.emailjs_template_id_email_only,
        template_id_dual=runtime_config.emailjs_template_id_dual,
        endpoint=runtime_config.emailjs_endpoint,
        timeout_seconds=runtime_config.emailjs_timeout_seconds,
    )

    # Initialize admin panel manager
    admin_manager = AdminPanelManager(
        auth_service,
        invite_notifier=invite_notifier,
        invite_login_url=runtime_config.invite_email_login_url,
    )

    projects_file_path, user_access_file_path, invites_file_path = _resolve_bootstrap_file_paths(runtime_config)
    state_store, supabase_validation_repository, state_backend_message = _build_supabase_state(runtime_config)
    bootstrap_warning = _bootstrap_auth_and_projects(
        auth_service,
        admin_manager,
        runtime_config,
        projects_file_path=str(projects_file_path),
        user_access_file_path=str(user_access_file_path),
        invites_file_path=str(invites_file_path),
        state_store=state_store,
    )

    def _current_project_map() -> dict[str, Project]:
        project_map: dict[str, Project] = {}
        for row in admin_manager.list_projects():
            slug = str(row.get("project_slug", "")).strip()
            if not slug:
                continue
            project = admin_manager.get_project(slug)
            if project is not None:
                project_map[slug] = project
        return project_map

    seed_warning = _validate_seed_file(runtime_config.detection_seed_path)
    seed_detections_by_project = _load_seed_detections(runtime_config.detection_seed_path)
    detection_repository = InMemoryDetectionRepository()
    queue_service = DetectionQueueService(detection_repository)
    service_ref: dict[str, DetectionQueueService] = {"queue": queue_service}
    loaded_project_signatures: dict[str, str] = {}
    loaded_project_warnings: dict[str, str] = {}
    loaded_project_order: list[str] = []
    max_loaded_projects = 3
    audio_service = AudioFetchService(EphemeralCacheManager(ttl_seconds=300, max_files=128))
    validation_repository = supabase_validation_repository or AppendOnlyValidationRepository(base_dir=runtime_config.validation_base_dir)
    validation_service = ValidationService(validation_repository)
    report_cache: dict[tuple[str, int, int, str], tuple[float, tuple[object, ...]]] = {}
    report_cache_ttl_seconds = 30.0

    with gr.Blocks(title="BirdNET-Validator-App", css=APP_CSS, elem_classes=["bn-shell"]) as wrapper:
        gr.HTML(app_header_html(state_backend_message))
        if state_backend_message.startswith("⚠"):
            gr.Markdown(state_backend_message)
        if bootstrap_warning:
            gr.Markdown(bootstrap_warning)

        # Session state
        session_state = gr.State(value=None)
        selected_project_state = gr.State(value=None)
        selected_dataset_repo_state = gr.State(value="")
        seed_warning_state = gr.State(value=seed_warning)

        def _project_rows() -> list[list[object]]:
            projects = admin_manager.list_projects()
            return [
                [
                    p["project_slug"],
                    p["name"],
                    p["dataset_repo_id"],
                    p.get("visibility", "collaborative"),
                    p.get("owner_username", ""),
                    "yes" if bool(p.get("dataset_token_set", False)) else "no",
                    "yes" if bool(p["active"]) else "no",
                ]
                for p in projects
            ]

        def _project_slugs() -> list[str]:
            return [p["project_slug"] for p in admin_manager.list_projects()]

        def _project_map() -> dict[str, Project]:
            result: dict[str, Project] = {}
            for slug in _project_slugs():
                project = admin_manager.get_project(slug)
                if project is not None:
                    result[slug] = project
            return result

        def _admin_projects_for_session(session) -> list[str]:
            if session is None:
                return []
            admin_projects: list[str] = []
            for project_slug in session.authorized_projects:
                role = auth_service.get_user_role_for_project(session.username, project_slug)
                if role == Role.admin:
                    admin_projects.append(project_slug)
            return sorted(admin_projects)

        def _is_admin_for_project(session, project_slug: str) -> bool:
            if session is None:
                return False
            slug = (project_slug or "").strip()
            if not slug:
                return False
            role = auth_service.get_user_role_for_project(session.username, slug)
            return role == Role.admin

        def _refresh_session_copy(session):
            if session is None:
                return None
            refreshed = auth_service.refresh_session_authorizations(session.session_id) or session
            return replace(refreshed, authorized_projects=list(refreshed.authorized_projects))

        def _project_queue_signature(project_slug: str, token: str | None) -> str:
            project = admin_manager.get_project(project_slug)
            token_digest = hashlib.sha1((token or "").encode("utf-8")).hexdigest() if token else ""
            seed_stamp = ""
            if runtime_config.detection_seed_path:
                seed_path = Path(runtime_config.detection_seed_path)
                try:
                    stat = seed_path.stat()
                    seed_stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
                except OSError:
                    seed_stamp = "missing"
            payload = {
                "project_slug": project_slug,
                "dataset_repo_id": project.dataset_repo_id if project is not None else "",
                "active": project.active if project is not None else False,
                "dataset_token": token_digest,
                "seed": seed_stamp,
                "demo": runtime_config.enable_demo_bootstrap,
            }
            return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

        def _touch_loaded_project(project_slug: str) -> None:
            if project_slug in loaded_project_order:
                loaded_project_order.remove(project_slug)
            loaded_project_order.append(project_slug)
            while len(loaded_project_order) > max_loaded_projects:
                stale_slug = loaded_project_order.pop(0)
                if stale_slug == project_slug:
                    continue
                detection_repository.remove_project(stale_slug)
                loaded_project_signatures.pop(stale_slug, None)
                loaded_project_warnings.pop(stale_slug, None)

        def _invalidate_report_cache(project_slug: str | None = None) -> None:
            if not project_slug:
                report_cache.clear()
                return
            for cache_key in list(report_cache.keys()):
                if cache_key[0] == project_slug:
                    report_cache.pop(cache_key, None)

        def _invalidate_project_queue(project_slug: str | None = None) -> None:
            if project_slug:
                detection_repository.remove_project(project_slug)
                loaded_project_signatures.pop(project_slug, None)
                loaded_project_warnings.pop(project_slug, None)
                if project_slug in loaded_project_order:
                    loaded_project_order.remove(project_slug)
                _invalidate_report_cache(project_slug)
                return

            for loaded_slug in list(loaded_project_signatures.keys()):
                detection_repository.remove_project(loaded_slug)
            loaded_project_signatures.clear()
            loaded_project_warnings.clear()
            loaded_project_order.clear()
            _invalidate_report_cache()

        def _ensure_project_queue_loaded(project_slug: str, session, force: bool = False) -> str:
            slug = (project_slug or "").strip()
            if not slug:
                return ""

            token = _project_fetch_token(slug, session)
            signature = _project_queue_signature(slug, token)
            if not force and loaded_project_signatures.get(slug) == signature:
                _touch_loaded_project(slug)
                return loaded_project_warnings.get(slug, "")

            items, warning = _project_detection_items(
                slug,
                seed_detections_by_project=seed_detections_by_project,
                project_map=_project_map(),
                allow_demo_defaults=runtime_config.enable_demo_bootstrap,
                hf_token=token,
            )
            detection_repository.seed(slug, items)
            loaded_project_signatures[slug] = signature
            loaded_project_warnings[slug] = warning
            _touch_loaded_project(slug)
            return warning

        def _persist_admin_state() -> tuple[bool, str]:
            try:
                _persist_bootstrap_state(
                    projects_path=projects_file_path,
                    user_access_path=user_access_file_path,
                    invites_path=invites_file_path,
                    admin_manager=admin_manager,
                    auth_service=auth_service,
                    state_store=state_store,
                )
                return True, ""
            except Exception as exc:
                return False, str(exc)

        with gr.Tabs(elem_classes=["bn-tabs"]):
            # ===== TAB 1: Login =====
            with gr.Tab("Login", id="login_tab"):
                gr.HTML(
                    section_header_html(
                        "Secure access",
                        "Sign in with your Hugging Face identity",
                        "Your account controls project access, validator attribution, and private dataset access.",
                    )
                )
                username_input, session_output, login_button, error_message = create_login_page(auth_service)

                # Store session ID when login succeeds
                def handle_login_success(session_id: str):
                    """Process successful login and store session."""
                    if session_id:
                        return auth_service.get_session(session_id)
                    return None

                session_output.change(
                    fn=handle_login_success,
                    inputs=[session_output],
                    outputs=[session_state],
                )

            # ===== TAB 2: Admin Panel =====
            with gr.Tab("Admin", id="admin_tab"):
                gr.HTML(
                    section_header_html(
                        "Administration",
                        "Manage projects, teams, invites, and private dataset access",
                        "Administrative actions are scoped by project role. Keep dataset tokens restricted to the projects that need them.",
                    )
                )
                admin_info = gr.Markdown(value="Login required")
                admin_scope_info = gr.Markdown(value="")
                admin_overview = gr.HTML(value=admin_overview_html(username=None, total_projects=0, admin_projects=0, validator_projects=0, pending_invites=0))

                def _render_admin_overview(session):
                    if session is None:
                        return admin_overview_html(
                            username=None,
                            total_projects=len(_project_rows()),
                            admin_projects=0,
                            validator_projects=0,
                            pending_invites=0,
                        )
                    admin_projects = _admin_projects_for_session(session)
                    validator_projects = [
                        project_slug
                        for project_slug in session.authorized_projects
                        if auth_service.get_user_role_for_project(session.username, project_slug) == Role.validator
                    ]
                    return admin_overview_html(
                        username=session.username,
                        total_projects=len(_project_rows()),
                        admin_projects=len(admin_projects),
                        validator_projects=len(validator_projects),
                        pending_invites=len(auth_service.list_pending_invites(session.username)),
                    )

                def create_admin_display(session):
                    """Show admin panel or access denied message."""
                    if session is None:
                        return (
                            "**Not authenticated** - Login first in the **Login** tab.",
                            gr.update(visible=False),
                        )
                    admin_projects = _admin_projects_for_session(session)
                    return (
                        (
                            f"**Admin workspace** - Welcome, {session.username}. "
                            f"You are admin in {len(admin_projects)} project(s). "
                            "You can always create a new project and become its admin."
                        ),
                        gr.update(visible=True),
                    )

                with gr.Column(visible=False, elem_classes=["bn-admin-section", "bn-admin-project-section"]) as admin_controls:
                    with gr.Group(elem_classes=["bn-admin-panel"]):
                        gr.HTML(
                            section_header_html(
                                "Projects",
                                "Project management",
                                "Create project records and review the datasets registered in this validator workspace.",
                                class_name="bn-panel-soft",
                            )
                        )
                        with gr.Row():
                            create_project_slug = gr.Textbox(
                                label="New Project Slug",
                                placeholder="ex: amazonas-2026",
                            )
                            create_project_name = gr.Textbox(
                                label="Project Name",
                                placeholder="ex: Amazonas Survey 2026",
                            )
                            create_project_repo = gr.Textbox(
                                label="HF Dataset Repo ID",
                                placeholder="ex: birdnet/amazonas-2026-dataset",
                            )
                            create_project_visibility = gr.Dropdown(
                                label="Visibility",
                                choices=["private", "collaborative"],
                                value="collaborative",
                            )
                            create_project_token = gr.Textbox(
                                label="Project HF Token (optional)",
                                placeholder="hf_xxx...",
                                type="password",
                            )

                        create_project_message = gr.Markdown()
                        projects_table = gr.Dataframe(
                            value=_project_rows(),
                            headers=["Project", "Name", "Dataset", "Visibility", "Owner", "Token", "Active"],
                            interactive=False,
                            elem_classes=["bn-dataframe"],
                        )

                    with gr.Row(elem_classes=["bn-admin-action-row"]):
                        create_project_btn = gr.Button(
                            "Create Project",
                            elem_classes=["bn-admin-action", "bn-admin-action-orange"],
                        )
                        refresh_projects_btn = gr.Button(
                            "Refresh List",
                            elem_classes=["bn-admin-action", "bn-admin-action-blue"],
                        )

                    def create_project(session, slug: str, name: str, repo_id: str, visibility: str, project_token: str):
                        if session is None:
                            return "Access denied. Login required.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), session

                        slug = (slug or "").strip()
                        name = (name or "").strip()
                        repo_id = (repo_id or "").strip()
                        visibility_value = (visibility or "collaborative").strip().lower()
                        project_token_value = (project_token or "").strip() or _session_hf_token(session) or None
                        if not slug or not name or not repo_id:
                            return "Fill slug, name, and repo id.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), session
                        if visibility_value not in {"private", "collaborative"}:
                            return "Visibility must be private or collaborative.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), session

                        created = admin_manager.register_project(
                            Project(
                                project_id=str(uuid4()),
                                project_slug=slug,
                                name=name,
                                dataset_repo_id=repo_id,
                                visibility=visibility_value,
                                owner_username=session.username,
                                dataset_token=project_token_value,
                                active=True,
                            )
                        )
                        if not created:
                            admin_projects = _admin_projects_for_session(session)
                            return (
                                f"Project '{slug}' already exists.",
                                _project_rows(),
                                gr.update(choices=admin_projects),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(choices=admin_projects),
                                gr.update(choices=admin_projects),
                                gr.update(choices=["all", *admin_projects], value="all"),
                                gr.update(),
                                session,
                            )

                        # Project creator is always admin of the project.
                        auth_service.upsert_user_project_role(session.username, slug, Role.admin)
                        persisted, persist_error = _persist_admin_state()
                        _invalidate_project_queue(slug)
                        refreshed_warning = ""
                        refreshed_session = _refresh_session_copy(session)
                        admin_projects = _admin_projects_for_session(refreshed_session)

                        return (
                            (
                                f"Project '{slug}' created successfully."
                                if persisted
                                else f"Project '{slug}' created, but could not persist bootstrap files: {persist_error}"
                            ),
                            _project_rows(),
                            gr.update(choices=admin_projects, value=slug),
                            gr.update(value=""),
                            gr.update(value=""),
                            gr.update(value=""),
                            gr.update(value="collaborative"),
                            gr.update(value=""),
                            gr.update(choices=admin_projects, value=slug),
                            gr.update(choices=admin_projects, value=slug),
                            gr.update(choices=["all", *admin_projects], value="all"),
                            refreshed_warning,
                            refreshed_session,
                        )

                    gr.HTML("<div class='bn-spacer'></div>")

                    def _render_admin_scope_info(session, selected_admin_project: str):
                        if session is None:
                            return ""

                        selected = (selected_admin_project or "").strip()
                        if not selected:
                            admin_projects = _admin_projects_for_session(session)
                            if not admin_projects:
                                return (
                                    "You are authenticated, but currently not admin of any existing project. "
                                    "Create a project to become admin of it."
                                )
                            return (
                                f"Select a project to manage. "
                                f"You are admin in {len(admin_projects)} project(s): {', '.join(admin_projects)}"
                            )

                        role = auth_service.get_user_role_for_project(session.username, selected)
                        if role == Role.admin:
                            return ""
                        if role == Role.validator:
                            return "Management actions require ADMIN for this project."
                        return f"You do not have access to project '{selected}'."

                    def refresh_projects(session):
                        if session is None:
                            return []
                        return _project_rows()

                    refresh_projects_btn.click(
                        fn=refresh_projects,
                        inputs=[session_state],
                        outputs=[projects_table],
                    )

                with gr.Column(visible=False, elem_classes=["bn-admin-section", "bn-admin-token-section"]) as admin_token_controls:
                    with gr.Group(elem_classes=["bn-admin-panel", "bn-admin-token-panel"]):
                        gr.HTML(
                            section_header_html(
                                "Access",
                                "Project token management",
                                "Store or clear dataset tokens without exposing them in the interface.",
                                class_name="bn-panel-soft",
                            )
                        )
                        with gr.Row():
                            token_project_select = gr.Dropdown(
                                choices=_project_slugs(),
                                label="Project",
                            )
                            token_new_value = gr.Textbox(
                                label="New token",
                                placeholder="hf_xxx...",
                                type="password",
                            )
                            token_clear_checkbox = gr.Checkbox(label="Clear token", value=False)
                        token_update_message = gr.Markdown()
                    token_update_btn = gr.Button(
                        "Update Project Token",
                        elem_classes=["bn-admin-action", "bn-admin-action-orange"],
                    )

                    def update_project_token(session, project_slug: str, new_token: str, clear_token: bool):
                        if session is None:
                            return "Access denied. Login required.", gr.update(), gr.update(), gr.update(), gr.update()
                        if not _is_admin_for_project(session, project_slug):
                            return "Access denied. You must be admin of the selected project.", gr.update(), gr.update(), gr.update(), gr.update()
                        project = admin_manager.get_project(project_slug)
                        if project is None:
                            return "Select a valid project.", gr.update(), gr.update(), gr.update(), gr.update()

                        if bool(clear_token):
                            project.dataset_token = None
                            message = f"Project token cleared for {project_slug}"
                        else:
                            candidate = (new_token or "").strip()
                            if not candidate:
                                return "Provide a token or select clear token.", gr.update(), gr.update(), gr.update(), gr.update()
                            project.dataset_token = candidate
                            message = f"Project token updated for {project_slug}"

                        persisted, persist_error = _persist_admin_state()
                        if not persisted:
                            message = f"{message} | Persistence failed: {persist_error}"

                        _invalidate_project_queue(project_slug)
                        refreshed_warning = ""

                        return message, gr.update(value=""), gr.update(value=False), _project_rows(), refreshed_warning

                    token_update_btn.click(
                        fn=update_project_token,
                        inputs=[session_state, token_project_select, token_new_value, token_clear_checkbox],
                        outputs=[token_update_message, token_new_value, token_clear_checkbox, projects_table, seed_warning_state],
                    )

                def delete_project(session, project_slug: str):
                    if session is None:
                        return "Access denied. Login required.", gr.update(), gr.update(), gr.update(), session, gr.update(), gr.update()
                    if not _is_admin_for_project(session, project_slug):
                        return "Access denied. You must be admin of the selected project.", gr.update(), gr.update(), gr.update(), session, gr.update(), gr.update()

                    success, msg = admin_manager.delete_project(session.username, project_slug)
                    if not success:
                        return msg, _project_rows(), gr.update(choices=_project_slugs()), gr.update(choices=_project_slugs()), session, gr.update(), gr.update()

                    persisted, persist_error = _persist_admin_state()
                    if not persisted:
                        msg = f"{msg} | Persistence failed: {persist_error}"

                    _invalidate_project_queue(project_slug)
                    refreshed_warning = ""
                    refreshed_session = _refresh_session_copy(session)

                    admin_projects = _admin_projects_for_session(refreshed_session)
                    return (
                        msg,
                        _project_rows(),
                        gr.update(choices=admin_projects, value=None),
                        gr.update(choices=admin_projects, value=None),
                        refreshed_session,
                        refreshed_warning,
                        gr.update(choices=admin_projects, value=None),
                    )

                with gr.Column(visible=False, elem_classes=["bn-admin-section", "bn-admin-delete-section"]) as admin_delete_controls:
                    with gr.Group(elem_classes=["bn-admin-panel", "bn-admin-delete-panel"]):
                        gr.HTML(
                            section_header_html(
                                "Project",
                                "Delete project",
                                "Remove a project from this validator workspace without deleting its Hugging Face dataset.",
                                class_name="bn-panel-soft",
                            )
                        )
                        gr.HTML(inline_hint_html("Deleting a project removes assignments and pending invites.", "danger"))
                        delete_project_slug = gr.Dropdown(
                            choices=_project_slugs(),
                            label="Project to delete",
                        )
                        delete_project_message = gr.Markdown()
                    delete_project_btn = gr.Button(
                        "Delete Project",
                        elem_classes=["bn-admin-action", "bn-admin-action-red", "bn-delete-project-action"],
                    )

                with gr.Column(visible=False, elem_classes=["bn-admin-section", "bn-admin-access-section"]) as admin_users_controls:
                    with gr.Group(elem_classes=["bn-admin-panel", "bn-admin-access-panel"]):
                        gr.HTML(
                            section_header_html(
                                "Team",
                                "User access and invitations",
                                "Assign known users directly or send invites when access should be accepted later.",
                                class_name="bn-panel-soft",
                            )
                        )
                        gr.HTML(
                            inline_hint_html(
                                "Assign immediately when you know the Hugging Face username. Use invites when you want the user to accept access later or receive email instructions."
                            )
                        )

                        invite_mode = gr.Radio(
                            choices=["Internal app only", "Email only", "Both"],
                            value="Both",
                            label="Invite Method",
                        )

                        with gr.Column():
                            admin_username = gr.Textbox(
                                label="HF Username (for internal app invite)",
                                placeholder="validator_001",
                                visible=True,
                            )
                            admin_invite_email = gr.Textbox(
                                label="Email Address (for email notification)",
                                placeholder="validator@example.org",
                                visible=True,
                            )

                        with gr.Row():
                            admin_project = gr.Dropdown(
                                choices=_project_slugs(),
                                label="Project",
                            )
                            admin_role = gr.Dropdown(
                                choices=["admin", "validator"],
                                value="validator",
                                label="Role",
                            )

                        admin_message = gr.Markdown()

                    # Update visibility based on invite mode
                    def update_invite_fields(mode: str):
                        if mode == "Internal app only":
                            return gr.update(visible=True), gr.update(visible=False)
                        elif mode == "Email only":
                            return gr.update(visible=False), gr.update(visible=True)
                        else:  # Both
                            return gr.update(visible=True), gr.update(visible=True)

                    invite_mode.change(
                        fn=update_invite_fields,
                        inputs=[invite_mode],
                        outputs=[admin_username, admin_invite_email],
                    )

                    with gr.Row(elem_classes=["bn-admin-action-row"]):
                        invite_btn = gr.Button(
                            "Send Invite",
                            elem_classes=["bn-admin-action", "bn-admin-action-orange"],
                        )
                        assign_btn = gr.Button(
                            "Assign",
                            elem_classes=["bn-admin-action", "bn-admin-action-blue"],
                        )

                    def assign_user(session, username: str, project: str, role: str):
                        if session is None:
                            return "Access denied. Login required.", gr.update(), gr.update(), gr.update(), gr.update()
                        if not _is_admin_for_project(session, project):
                            return "Access denied. You must be admin of the selected project.", gr.update(), gr.update(), gr.update(), gr.update()
                        success, msg = admin_manager.assign_user_to_project(
                            session.username,
                            username,
                            project,
                            role,
                        )
                        if success:
                            persisted, persist_error = _persist_admin_state()
                            final_message = msg if persisted else f"{msg} | Persistence failed: {persist_error}"
                            return final_message, gr.update(value=""), gr.update(value=""), gr.update(value=None), gr.update(value="validator")
                        return msg, gr.update(), gr.update(), gr.update(), gr.update()

                    assign_btn.click(
                        fn=assign_user,
                        inputs=[session_state, admin_username, admin_project, admin_role],
                        outputs=[admin_message, admin_username, admin_invite_email, admin_project, admin_role],
                    )

                    def invite_user(session, mode: str, username: str, invite_email: str, project: str, role: str):
                        if session is None:
                            return "Access denied. Login required.", gr.update(), gr.update(), gr.update(), gr.update()
                        if not _is_admin_for_project(session, project):
                            return "Access denied. You must be admin of the selected project.", gr.update(), gr.update(), gr.update(), gr.update()

                        final_username = None if mode == "Email only" else (username or None)
                        final_email = None if mode == "Internal app only" else (invite_email or None)

                        success, msg = admin_manager.invite_user_to_project(
                            actor_username=session.username,
                            invited_by=session.username,
                            username=final_username,
                            invitee_email=final_email,
                            project_slug=project,
                            role=role,
                        )
                        if success:
                            persisted, persist_error = _persist_admin_state()
                            final_message = msg if persisted else f"{msg} | Persistence failed: {persist_error}"
                            return final_message, gr.update(value=""), gr.update(value=""), gr.update(value=None), gr.update(value="validator")
                        return msg, gr.update(), gr.update(), gr.update(), gr.update()

                    invite_event = invite_btn.click(
                        fn=invite_user,
                        inputs=[session_state, invite_mode, admin_username, admin_invite_email, admin_project, admin_role],
                        outputs=[admin_message, admin_username, admin_invite_email, admin_project, admin_role],
                    )

                with gr.Column(visible=False, elem_classes=["bn-admin-section", "bn-admin-pending-section"]) as admin_pending_controls:
                    with gr.Group(elem_classes=["bn-admin-panel", "bn-admin-pending-panel"]):
                        gr.HTML(
                            section_header_html(
                                "Invites",
                                "Pending access requests",
                                "Review outstanding invitations and revoke stale access before onboarding more validators.",
                                class_name="bn-panel-soft",
                            )
                        )
                        with gr.Group(elem_classes=["bn-card-body"]):
                            with gr.Row():
                                pending_invites_filter_project = gr.Dropdown(
                                    choices=["all", *_project_slugs()],
                                    value="all",
                                    label="Filter by project",
                                )
                                pending_invite_username = gr.Textbox(label="Invite username", placeholder="validator_001")
                                pending_invite_project = gr.Dropdown(choices=_project_slugs(), label="Invite project")
                            pending_invites_table = gr.Dataframe(
                                value=[],
                                headers=["Username", "Project", "Role", "Invited by", "Expires at", "Expires in"],
                                interactive=False,
                                elem_classes=["bn-dataframe"],
                            )
                            pending_invites_message = gr.Markdown()
                    with gr.Row(elem_classes=["bn-admin-action-row"]):
                        refresh_pending_invites_btn = gr.Button(
                            "Refresh pending invites",
                            elem_classes=["bn-admin-action", "bn-admin-action-orange"],
                        )
                        revoke_invite_btn = gr.Button(
                            "Revoke Invite",
                            elem_classes=["bn-admin-action", "bn-admin-action-blue"],
                        )

                    def _pending_invites_rows(project_filter: str, session):
                        def _remaining_from_iso(iso_value: str) -> str:
                            raw = str(iso_value or "").strip()
                            if not raw:
                                return "unknown"
                            try:
                                expires_at = datetime.fromisoformat(raw)
                            except Exception:
                                return "unknown"
                            if expires_at.tzinfo is None:
                                now = datetime.now()
                            else:
                                now = datetime.now(expires_at.tzinfo)
                            remaining_seconds = int((expires_at - now).total_seconds())
                            if remaining_seconds <= 0:
                                return "expired"
                            days = remaining_seconds // 86400
                            hours = (remaining_seconds % 86400) // 3600
                            minutes = (remaining_seconds % 3600) // 60
                            if days > 0:
                                return f"{days}d {hours}h"
                            if hours > 0:
                                return f"{hours}h {minutes}m"
                            return f"{minutes}m"

                        selected = (project_filter or "all").strip().lower()
                        project = None if selected == "all" else project_filter
                        if project is not None and not _is_admin_for_project(session, project):
                            return []

                        admin_scope = set(_admin_projects_for_session(session))
                        invites = admin_manager.list_pending_invites(project_slug=project)
                        return [
                            [
                                row.get("username", ""),
                                row.get("project_slug", ""),
                                row.get("role", ""),
                                row.get("invited_by", ""),
                                row.get("expires_at", ""),
                                _remaining_from_iso(str(row.get("expires_at", ""))),
                            ]
                            for row in invites
                            if str(row.get("project_slug", "")) in admin_scope
                        ]

                    refresh_pending_invites_btn.click(
                        fn=_pending_invites_rows,
                        inputs=[pending_invites_filter_project, session_state],
                        outputs=[pending_invites_table],
                    )

                    invite_event.then(
                        fn=_pending_invites_rows,
                        inputs=[pending_invites_filter_project, session_state],
                        outputs=[pending_invites_table],
                    )

                    def revoke_invite(session, username: str, project_slug: str, project_filter: str):
                        if session is None:
                            return "Access denied. Login required.", _pending_invites_rows(project_filter, session)
                        if not _is_admin_for_project(session, project_slug):
                            return "Access denied. You must be admin of the selected project.", _pending_invites_rows(project_filter, session)
                        success, msg = admin_manager.revoke_invite(username=username, project_slug=project_slug)
                        if success:
                            persisted, persist_error = _persist_admin_state()
                            if not persisted:
                                msg = f"{msg} | Persistence failed: {persist_error}"
                        return msg, _pending_invites_rows(project_filter, session)

                    revoke_invite_btn.click(
                        fn=revoke_invite,
                        inputs=[session_state, pending_invite_username, pending_invite_project, pending_invites_filter_project],
                        outputs=[pending_invites_message, pending_invites_table],
                    )

                    pending_invites_filter_project.change(
                        fn=_pending_invites_rows,
                        inputs=[pending_invites_filter_project, session_state],
                        outputs=[pending_invites_table],
                    )

                    delete_project_btn.click(
                        fn=delete_project,
                        inputs=[session_state, delete_project_slug],
                        outputs=[
                            delete_project_message,
                            projects_table,
                            admin_project,
                            token_project_select,
                            session_state,
                            seed_warning_state,
                            delete_project_slug,
                        ],
                    )

                create_project_event = create_project_btn.click(
                    fn=create_project,
                    inputs=[session_state, create_project_slug, create_project_name, create_project_repo, create_project_visibility, create_project_token],
                    outputs=[
                        create_project_message,
                        projects_table,
                        admin_project,
                        create_project_slug,
                        create_project_name,
                        create_project_repo,
                        create_project_visibility,
                        create_project_token,
                        token_project_select,
                        pending_invite_project,
                        pending_invites_filter_project,
                        seed_warning_state,
                        session_state,
                    ],
                )

                session_state.change(
                    fn=create_admin_display,
                    inputs=[session_state],
                    outputs=[admin_info, admin_controls],
                )

                session_state.change(
                    fn=_render_admin_overview,
                    inputs=[session_state],
                    outputs=[admin_overview],
                )

                session_state.change(
                    fn=_render_admin_scope_info,
                    inputs=[session_state, admin_project],
                    outputs=[admin_scope_info],
                )

                session_state.change(
                    fn=lambda s: (
                        gr.update(visible=bool(s is not None)),
                        gr.update(visible=bool(s is not None)),
                        gr.update(visible=bool(s is not None)),
                        gr.update(visible=bool(s is not None)),
                    ),
                    inputs=[session_state],
                    outputs=[admin_token_controls, admin_delete_controls, admin_users_controls, admin_pending_controls],
                )

                session_state.change(
                    fn=lambda s: _project_rows() if s is not None else [],
                    inputs=[session_state],
                    outputs=[projects_table],
                )

                session_state.change(
                    fn=lambda s: (
                        gr.update(choices=_admin_projects_for_session(s), value=(_admin_projects_for_session(s)[0] if _admin_projects_for_session(s) else None)),
                        gr.update(choices=_admin_projects_for_session(s), value=(_admin_projects_for_session(s)[0] if _admin_projects_for_session(s) else None)),
                        gr.update(choices=_admin_projects_for_session(s), value=(_admin_projects_for_session(s)[0] if _admin_projects_for_session(s) else None)),
                        gr.update(choices=_admin_projects_for_session(s), value=(_admin_projects_for_session(s)[0] if _admin_projects_for_session(s) else None)),
                        gr.update(choices=["all", *_admin_projects_for_session(s)], value="all"),
                        [],
                    ),
                    inputs=[session_state],
                    outputs=[admin_project, token_project_select, pending_invite_project, delete_project_slug, pending_invites_filter_project, pending_invites_table],
                )

                admin_project.change(
                    fn=_render_admin_scope_info,
                    inputs=[session_state, admin_project],
                    outputs=[admin_scope_info],
                )

            # ===== TAB 3: Project Selection =====
            with gr.Tab("Projects", id="project_tab"):
                gr.HTML(
                    section_header_html(
                        "Project context",
                        "Choose the dataset you want to validate",
                        "Project access is filtered by your role. Invitations can be accepted here before validation starts.",
                    )
                )
                project_overview = gr.HTML(value=project_overview_html([], []), visible=False)
                project_info_display = gr.Markdown(
                    value="Login first in the **Login** tab"
                )
                project_context_display = gr.HTML(value=project_context_html(None))
                project_selector = gr.Dropdown(
                    choices=[],
                    label="Authorized Project",
                    interactive=False,
                    allow_custom_value=True,
                )
                invitations_info = gr.Markdown(value="")
                invitations_overview = gr.HTML(value=invite_panel_html(0))
                invite_selector = gr.Dropdown(choices=[], label="Pending Invites", interactive=False)
                with gr.Row():
                    refresh_invites_btn = gr.Button("Refresh invites", elem_classes=["bn-soft-action"])
                    accept_invite_btn = gr.Button("Accept Invite", elem_classes=["bn-soft-action"])
                    accept_all_invites_btn = gr.Button("Accept All", elem_classes=["bn-soft-action"])
                    reject_invite_btn = gr.Button("Reject Invite", elem_classes=["bn-soft-action"])

                def update_project_selector(session):
                    """Update project dropdown when user logs in."""
                    if session is None:
                        return (
                            gr.update(choices=[], value=None, interactive=False),
                            project_overview_html([], []),
                            "Not authenticated. Login first.",
                            project_context_html(None),
                            None,
                            "",
                        )

                    projects = session.authorized_projects
                    if not projects:
                        return (
                            gr.update(choices=[], value=None, interactive=False),
                            project_overview_html(_project_rows(), []),
                            (
                                "**No projects available yet**\n\n"
                                "To get started:\n"
                                "1. Go to the **Admin** tab.\n"
                                "2. Fill **New Project Slug**, **Project Name**, and **HF Dataset Repo ID**.\n"
                                "3. Click **Create Project**.\n"
                                "4. Go back to **Projects** and choose the created project."
                            ),
                            project_context_html(None),
                            None,
                            "",
                        )

                    selected = projects[0]
                    role = auth_service.get_user_role_for_project(session.username, selected)
                    role_label = role.value.upper() if role else "UNKNOWN"
                    selected_project = admin_manager.get_project(selected)
                    dataset_repo_id = selected_project.dataset_repo_id if selected_project else ""
                    project_row = next((row for row in _project_rows() if row and str(row[0]) == selected), None)
                    return (
                        gr.update(choices=projects, value=selected, interactive=True),
                        project_overview_html(_project_rows(), projects, selected),
                        "",
                        project_context_html(project_row, role_label),
                        selected,
                        dataset_repo_id,
                    )

                def _format_invite_option(invite) -> str:
                    return f"{invite.project_slug}|{invite.role.value}|{invite.invited_by}|{invite.expires_at.isoformat()}"

                def _format_invite_remaining(expires_at: datetime) -> str:
                    now = datetime.now(expires_at.tzinfo) if expires_at.tzinfo else datetime.utcnow()
                    delta = expires_at - now
                    remaining_seconds = int(delta.total_seconds())
                    if remaining_seconds <= 0:
                        return "expired"

                    days = remaining_seconds // 86400
                    hours = (remaining_seconds % 86400) // 3600
                    minutes = (remaining_seconds % 3600) // 60

                    if days > 0:
                        return f"{days}d {hours}h"
                    if hours > 0:
                        return f"{hours}h {minutes}m"
                    return f"{minutes}m"

                def _build_invite_label(invite) -> str:
                    remaining = _format_invite_remaining(invite.expires_at)
                    return (
                        f"{invite.project_slug} ({invite.role.value})"
                        f" - invited by {invite.invited_by}"
                        f" - expires in {remaining}"
                    )

                def _build_invites_ui(session):
                    if session is None:
                        return gr.update(value="", visible=False), invite_panel_html(0), gr.update(choices=[], value=None, interactive=False)
                    invites = auth_service.list_pending_invites(session.username)
                    if not invites:
                        return gr.update(value="No pending invites", visible=True), invite_panel_html(0), gr.update(choices=[], value=None, interactive=False)
                    encoded = [_format_invite_option(item) for item in invites]
                    labeled_choices = [(_build_invite_label(invite), encoded_value) for invite, encoded_value in zip(invites, encoded)]
                    return (
                        gr.update(value=f"Pending invites: {len(labeled_choices)}", visible=True),
                        invite_panel_html(len(labeled_choices)),
                        gr.update(choices=labeled_choices, value=encoded[0], interactive=True),
                    )

                def _parse_invite_option(raw_option: str) -> tuple[str, str, str]:
                    """Compatibility parser: supports old and new encoded invite strings."""
                    value = str(raw_option or "").strip()
                    parts = value.split("|")
                    if len(parts) < 3:
                        return "", "", ""
                    return parts[0].strip(), parts[1].strip(), parts[2].strip()

                def _accept_invite(session, selected_option: str):
                    if session is None:
                        return "Login first", session
                    project_slug, _, _ = _parse_invite_option(selected_option)
                    if not project_slug:
                        return "Select an invite", session
                    success, message = auth_service.accept_project_invite(session.username, project_slug)
                    refreshed = auth_service.refresh_session_authorizations(session.session_id) or session
                    if success:
                        _persist_admin_state()
                    return message, refreshed

                def _reject_invite(session, selected_option: str):
                    if session is None:
                        return "Login first", session
                    project_slug, _, _ = _parse_invite_option(selected_option)
                    if not project_slug:
                        return "Select an invite", session
                    success, message = auth_service.reject_project_invite(session.username, project_slug)
                    refreshed = auth_service.refresh_session_authorizations(session.session_id) or session
                    if success:
                        _persist_admin_state()
                    return message, refreshed

                def _accept_all_invites(session):
                    if session is None:
                        return "Login first", session
                    accepted, failed, message = auth_service.accept_all_project_invites(session.username)
                    refreshed = auth_service.refresh_session_authorizations(session.session_id) or session
                    if accepted > 0:
                        _persist_admin_state()
                    detail = f"{message}"
                    if failed:
                        detail = f"{detail} | failed={failed}"
                    return detail, refreshed

                session_state.change(
                    fn=update_project_selector,
                    inputs=[session_state],
                    outputs=[project_selector, project_overview, project_info_display, project_context_display, selected_project_state, selected_dataset_repo_state],
                )

                session_state.change(
                    fn=_build_invites_ui,
                    inputs=[session_state],
                    outputs=[invitations_info, invitations_overview, invite_selector],
                )

                refresh_invites_btn.click(
                    fn=_build_invites_ui,
                    inputs=[session_state],
                    outputs=[invitations_info, invitations_overview, invite_selector],
                )

                accept_invite_btn.click(
                    fn=_accept_invite,
                    inputs=[session_state, invite_selector],
                    outputs=[project_info_display, session_state],
                ).then(
                    fn=update_project_selector,
                    inputs=[session_state],
                    outputs=[project_selector, project_overview, project_info_display, project_context_display, selected_project_state, selected_dataset_repo_state],
                ).then(
                    fn=_build_invites_ui,
                    inputs=[session_state],
                    outputs=[invitations_info, invitations_overview, invite_selector],
                )

                accept_all_invites_btn.click(
                    fn=_accept_all_invites,
                    inputs=[session_state],
                    outputs=[project_info_display, session_state],
                ).then(
                    fn=update_project_selector,
                    inputs=[session_state],
                    outputs=[project_selector, project_overview, project_info_display, project_context_display, selected_project_state, selected_dataset_repo_state],
                ).then(
                    fn=_build_invites_ui,
                    inputs=[session_state],
                    outputs=[invitations_info, invitations_overview, invite_selector],
                )

                reject_invite_btn.click(
                    fn=_reject_invite,
                    inputs=[session_state, invite_selector],
                    outputs=[project_info_display, session_state],
                ).then(
                    fn=_build_invites_ui,
                    inputs=[session_state],
                    outputs=[invitations_info, invitations_overview, invite_selector],
                )

                def update_selected_project(selected: str, session):
                    """Update state when project is selected."""
                    if session and selected:
                        selected_project = admin_manager.get_project(selected)
                        dataset_repo_id = selected_project.dataset_repo_id if selected_project else ""
                        role = auth_service.get_user_role_for_project(session.username, selected)
                        role_label = role.value.upper() if role else "UNKNOWN"
                        project_row = next((row for row in _project_rows() if row and str(row[0]) == selected), None)
                        return selected, dataset_repo_id, project_overview_html(_project_rows(), session.authorized_projects, selected), project_context_html(project_row, role_label)
                    return None, "", project_overview_html([], []), project_context_html(None)

                project_selector.change(
                    fn=update_selected_project,
                    inputs=[project_selector, session_state],
                    outputs=[selected_project_state, selected_dataset_repo_state, project_overview, project_context_display],
                )

                create_project_event.then(
                    fn=update_project_selector,
                    inputs=[session_state],
                    outputs=[project_selector, project_overview, project_info_display, project_context_display, selected_project_state, selected_dataset_repo_state],
                ).then(
                    fn=_build_invites_ui,
                    inputs=[session_state],
                    outputs=[invitations_info, invitations_overview, invite_selector],
                ).then(
                    fn=_render_admin_overview,
                    inputs=[session_state],
                    outputs=[admin_overview],
                ).then(
                    fn=create_admin_display,
                    inputs=[session_state],
                    outputs=[admin_info, admin_controls],
                )

            # ===== TAB 4: Validation =====
            with gr.Tab("Validate", id="validation_tab"):
                gr.HTML(
                    section_header_html(
                        "Validation workbench",
                        "Review audio segments",
                        "Load a project, listen to the current segment, review the spectrogram, and keep the queue moving with clear status actions.",
                    )
                )
                validation_status = gr.Markdown(
                    value="",
                    visible=False,
                    elem_classes=["bn-status-strip"],
                )
                queue_badge = gr.HTML(value="", visible=False)
                seed_warning_banner = gr.Markdown(value="", visible=False)

                def render_seed_warning(warning_text: str):
                    text = (warning_text or "").strip()
                    if not text:
                        return gr.update(value="", visible=False)
                    return gr.update(value=text, visible=True)

                seed_warning_state.change(
                    fn=render_seed_warning,
                    inputs=[seed_warning_state],
                    outputs=[seed_warning_banner],
                )
                wrapper.load(
                    fn=render_seed_warning,
                    inputs=[seed_warning_state],
                    outputs=[seed_warning_banner],
                )

                def get_validation_status(session, selected_project, dataset_repo_id):
                    """Show status message based on login/project state."""
                    if session is None:
                        return "**Not authenticated** - Login first in the **Login** tab"
                    if selected_project is None:
                        return "**Project not selected** - Select a project in the **Projects** tab"
                    total_detections = _get_project_detection_count(service_ref["queue"], selected_project)
                    return (
                        f"**Ready to validate** - Project: **{selected_project}** | "
                        f"User: **{session.username}** | Dataset: **{dataset_repo_id or 'not set'}** | "
                        f"Loaded detections: **{total_detections}**"
                    )

                session_state.change(
                    fn=lambda s, p, r: get_validation_status(s, p, r),
                    inputs=[session_state, selected_project_state, selected_dataset_repo_state],
                    outputs=[validation_status],
                )
                session_state.change(
                    fn=lambda p: _build_queue_badge(service_ref["queue"], p),
                    inputs=[selected_project_state],
                    outputs=[queue_badge],
                )

                selected_project_state.change(
                    fn=lambda s, p, r: get_validation_status(s, p, r),
                    inputs=[session_state, selected_project_state, selected_dataset_repo_state],
                    outputs=[validation_status],
                )
                selected_project_state.change(
                    fn=lambda p: _build_queue_badge(service_ref["queue"], p),
                    inputs=[selected_project_state],
                    outputs=[queue_badge],
                )

                selected_dataset_repo_state.change(
                    fn=lambda s, p, r: get_validation_status(s, p, r),
                    inputs=[session_state, selected_project_state, selected_dataset_repo_state],
                    outputs=[validation_status],
                )

                wrapper.load(
                    fn=lambda p: _build_queue_badge(service_ref["queue"], p),
                    inputs=[selected_project_state],
                    outputs=[queue_badge],
                )

                page_state = gr.State(value=1)
                project_species_state = gr.State(value=[])
                custom_corrected_species_state = gr.State(value={})
                favorite_detection_state = gr.State(value={})

                with gr.Row(equal_height=False, elem_classes=["bn-validation-grid"]):
                    with gr.Column(scale=8, elem_classes=["bn-media-panel"]):
                        validation_summary_cards = gr.HTML(value=_build_validation_summary_cards([]))
                        selected_segment_card = gr.HTML(value=selected_segment_html(None))

                        spectrogram_title = gr.Markdown(_spectrogram_title(None, None))
                        spectrogram_image = gr.Image(
                            label="",
                            type="filepath",
                            interactive=False,
                            height=330,
                        )
                        with gr.Row():
                            audio_player = gr.Audio(label="Selected audio", type="filepath", autoplay=True)
                        auto_play_audio = gr.Checkbox(label="Auto-play when selecting a row", value=True)

                        with gr.Row(elem_classes=["bn-action-row"]):
                            approve_btn = gr.Button("Confirm", variant="primary")
                            reject_btn = gr.Button("Reject")
                            uncertain_btn = gr.Button("Uncertain")
                            skip_btn = gr.Button("Skip")
                            favorite_btn = gr.Button("Favorite", variant="secondary")

                        corrected_species_input = gr.Dropdown(
                            label="Corrected species",
                            choices=["Noise", "Undetermined"],
                            allow_custom_value=True,
                            filterable=True,
                            value=None,
                        )

                        status = gr.Markdown(value="", elem_classes=["bn-status-strip"])

                        table = gr.Dataframe(
                            headers=[
                                "Key",
                                "Audio",
                                "Species",
                                "Confidence",
                                "Start",
                                "End",
                                "Status",
                                "Version",
                                "Conflict",
                                "Severity",
                            ],
                            label="Validation queue",
                            interactive=False,
                            elem_classes=["bn-dataframe"],
                        )
                        selected_index = gr.Number(label="Selected row", value=0, precision=0, visible=False)

                    with gr.Column(scale=4, elem_classes=["bn-sidebar-panel"]):
                        dataset_repo = gr.Textbox(label="Dataset repo", interactive=False)
                        species_filter = gr.Dropdown(
                            label="Species",
                            choices=[],
                            value=None,
                            interactive=False,
                        )

                        gr.Markdown("### Queue navigation")
                        with gr.Row():
                            prev_btn = gr.Button("Previous")
                            next_btn = gr.Button("Next")

                        with gr.Group(elem_classes=["bn-filter-panel"]):
                            gr.Markdown("### Filters")
                            min_confidence = gr.Slider(label="Minimum confidence", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                            validation_status_filter = gr.Dropdown(
                                label="Status",
                                choices=["all", "pending", "positive", "negative", "uncertain", "skip"],
                                value="all",
                            )
                            validator_filter = gr.Textbox(label="Validator", placeholder="Ex: validator_001")
                            updated_after_filter = gr.DateTime(label="Updated since", include_time=False, type="string")
                            show_conflicts_only = gr.Checkbox(label="Show only conflicts", value=False)
                            refresh_btn = gr.Button("Apply filters", variant="primary")

                        gr.Markdown("### Review details")
                        validator_name = gr.Textbox(label="Validator", value="", interactive=False)
                        validation_notes = gr.Textbox(label="Notes", placeholder="Optional", lines=4)
                        keyboard_shortcuts_info = gr.HTML(
                            value="<script>"
                            "document.addEventListener('keydown', function(event) {"
                            "  if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') return;"
                            "  const key = event.key;"
                            "  let buttonText = null;"
                            "  if (key === 'ArrowUp' || key === '1') buttonText = 'Confirm';"
                            "  else if (key === 'ArrowDown' || key === '2') buttonText = 'Reject';"
                            "  else if (key === '3') buttonText = 'Uncertain';"
                            "  else if (key === '4') buttonText = 'Skip';"
                            "  if (!buttonText) return;"
                            "  event.preventDefault();"
                            "  const buttons = document.querySelectorAll('button');"
                            "  for (const btn of buttons) {"
                            "    if ((btn.textContent || '').includes(buttonText)) { btn.click(); break; }"
                            "  }"
                            "});"
                            "</script>"
                        )

                cache_key_state = gr.State(value="")
                pending_status_state = gr.State(value="")
                conflict_detection_key_state = gr.State(value="")
                def _session_hf_token(session) -> str | None:
                    if session is None:
                        return None
                    return auth_service.get_hf_token_for_user(session.username)

                def _project_fetch_token(project_slug: str, session) -> str | None:
                    session_token = _session_hf_token(session)
                    project = admin_manager.get_project(project_slug) if project_slug else None
                    return _resolve_project_fetch_token(project, session_token)

                def refresh(
                    project_slug: str,
                    page: int,
                    species: str,
                    confidence: float,
                    validator_filter_value: str,
                    status_filter_value: str,
                    updated_after_value: object,
                    only_conflicts: bool,
                ):
                    if not project_slug:
                        return [], "", 1
                    species_name = (species or "").strip()
                    if not species_name:
                        return [], "Select a species to start validation", 1

                    rows, status_text, updated_page = _page_to_table(
                        service=service_ref["queue"],
                        snapshot_reader=validation_repository,
                        project_slug=project_slug,
                        page=page,
                        scientific_name=species_name,
                        min_confidence=confidence,
                        page_size=10,
                        validator_filter=validator_filter_value,
                        status_filter=status_filter_value,
                        updated_after=updated_after_value,
                        show_conflicts_only=only_conflicts,
                    )
                    rows = _sort_rows_by_confidence_desc(rows)
                    return rows, status_text, updated_page

                def go_next(
                    project_slug: str,
                    page: int,
                    species: str,
                    confidence: float,
                    validator_filter_value: str,
                    status_filter_value: str,
                    updated_after_value: object,
                    only_conflicts: bool,
                ):
                    return refresh(
                        project_slug,
                        page + 1,
                        species,
                        confidence,
                        validator_filter_value,
                        status_filter_value,
                        updated_after_value,
                        only_conflicts,
                    )

                def go_prev(
                    project_slug: str,
                    page: int,
                    species: str,
                    confidence: float,
                    validator_filter_value: str,
                    status_filter_value: str,
                    updated_after_value: object,
                    only_conflicts: bool,
                ):
                    return refresh(
                        project_slug,
                        max(1, page - 1),
                        species,
                        confidence,
                        validator_filter_value,
                        status_filter_value,
                        updated_after_value,
                        only_conflicts,
                    )

                def refresh_for_selected_project(project_slug: str, session):
                    if not project_slug:
                        return gr.update(choices=[], value=None, interactive=False), [], "", 1, None, None, _spectrogram_title(None, None), _build_validation_summary_cards([]), gr.update(choices=["Noise", "Undetermined"], value=None), []

                    warning = _ensure_project_queue_loaded(project_slug, session)

                    species_options = _extract_species_options_from_queue(
                        queue_service=service_ref["queue"],
                        project_slug=project_slug,
                        page_size=max(32, runtime_config.page_size),
                    )
                    corrected_choices = species_options + ["Noise", "Undetermined"]
                    return (
                        gr.update(choices=species_options, value=None, interactive=True),
                        [],
                        warning or "Select a species to start validation",
                        1,
                        None,
                        None,
                        _spectrogram_title(None, None),
                        _build_validation_summary_cards([]),
                        gr.update(choices=corrected_choices, value=None),
                        species_options,
                    )

                def save_for_project(
                    project_slug: str,
                    status_value: str,
                    rows: object,
                    idx: int,
                    session,
                    notes: str,
                    corrected_species_value: str | None,
                    cache_key: str,
                    page: int,
                    species: str,
                    confidence: float,
                    validator_filter_value: str,
                    status_filter_value: str,
                    updated_after_value: object,
                    only_conflicts: bool,
                ):
                    if not project_slug:
                        return "Select a project before validating", cache_key, None, rows, page, idx, "", ""
                    validator_name_value = _validator_name_from_session(session)
                    if not validator_name_value:
                        return "Login before validating", cache_key, None, rows, page, idx, "", ""
                    result = _save_selected_validation_with_refresh(
                        validation_service=validation_service,
                        audio_service=audio_service,
                        queue_service=service_ref["queue"],
                        snapshot_reader=validation_repository,
                        project_slug=project_slug,
                        rows=rows,
                        selected_index=int(idx),
                        status_value=status_value,
                        validator=validator_name_value,
                        notes=notes,
                        corrected_species=corrected_species_value,
                        cache_key=cache_key,
                        page=int(page),
                        scientific_name=species,
                        min_confidence=float(confidence),
                        validator_filter=validator_filter_value,
                        status_filter=status_filter_value,
                        updated_after=updated_after_value,
                        show_conflicts_only=bool(only_conflicts),
                    )
                    _invalidate_report_cache(project_slug)
                    return result

                def reapply_for_project(
                    project_slug: str,
                    rows: object,
                    idx: int,
                    pending_status: str,
                    conflict_key: str,
                    name: str,
                    notes: str,
                    cache_key: str,
                    page: int,
                    species: str,
                    confidence: float,
                    validator_filter_value: str,
                    status_filter_value: str,
                    updated_after_value: object,
                    only_conflicts: bool,
                ):
                    if not project_slug:
                        return "Select a project before reapplying", cache_key, None, rows, page, idx, pending_status, conflict_key
                    result = _reapply_last_conflict_validation_with_refresh(
                        validation_service=validation_service,
                        audio_service=audio_service,
                        queue_service=service_ref["queue"],
                        snapshot_reader=validation_repository,
                        project_slug=project_slug,
                        rows=rows,
                        selected_index=int(idx),
                        pending_status_value=pending_status,
                        conflict_detection_key=conflict_key,
                        validator=name,
                        notes=notes,
                        cache_key=cache_key,
                        page=int(page),
                        scientific_name=species,
                        min_confidence=float(confidence),
                        validator_filter=validator_filter_value,
                        status_filter=status_filter_value,
                        updated_after=updated_after_value,
                        show_conflicts_only=bool(only_conflicts),
                    )
                    _invalidate_report_cache(project_slug)
                    return result

                def batch_for_project(
                    project_slug: str,
                    rows: object,
                    status_value: str,
                    name: str,
                    notes: str,
                    cache_key: str,
                    page: int,
                    species: str,
                    confidence: float,
                    validator_filter_value: str,
                    status_filter_value: str,
                    updated_after_value: object,
                ):
                    if not project_slug:
                        return "Select a project before validating", cache_key, None, rows, page
                    result = _batch_validate_conflicts(
                        validation_service=validation_service,
                        audio_service=audio_service,
                        queue_service=service_ref["queue"],
                        snapshot_reader=validation_repository,
                        project_slug=project_slug,
                        rows=rows,
                        status_value=status_value,
                        validator=name,
                        notes=notes,
                        cache_key=cache_key,
                        page=int(page),
                        scientific_name=species,
                        min_confidence=float(confidence),
                        validator_filter=validator_filter_value,
                        status_filter=status_filter_value,
                        updated_after=updated_after_value,
                    )
                    _invalidate_report_cache(project_slug)
                    return result

                def build_report_for_project(project_slug: str) -> str:
                    if not project_slug:
                        return "Select a project to generate report"
                    return _build_validation_report(validation_repository, project_slug)

                def save_corrected_species_option(
                    project_slug: str,
                    corrected_value: str | None,
                    detected_species: list[str],
                    custom_by_project: dict[str, list[str]],
                ):
                    base_choices = list(dict.fromkeys([*detected_species, "Noise", "Undetermined"]))
                    value = (corrected_value or "").strip()

                    if not project_slug:
                        return gr.update(choices=base_choices, value=value or None), custom_by_project

                    updated = {k: list(v) for k, v in (custom_by_project or {}).items()}
                    custom_values = updated.get(project_slug, [])
                    if value and value not in base_choices and value not in custom_values:
                        custom_values.append(value)
                        updated[project_slug] = custom_values

                    final_choices = list(dict.fromkeys([*base_choices, *updated.get(project_slug, [])]))
                    return gr.update(choices=final_choices, value=value or None), updated

                def toggle_favorite_detection(
                    project_slug: str,
                    rows: object,
                    idx: int,
                    favorite_map: dict[str, list[str]],
                ):
                    normalized_rows = _normalize_rows(rows)
                    if not project_slug or not normalized_rows:
                        return "No detection selected to favorite", favorite_map, gr.update(value="Favorite", variant="secondary")

                    safe_idx = max(0, min(int(idx), len(normalized_rows) - 1))
                    detection_key = str(normalized_rows[safe_idx][0]).strip()
                    updated_map = {k: list(v) for k, v in (favorite_map or {}).items()}
                    project_favs = set(updated_map.get(project_slug, []))
                    if detection_key in project_favs:
                        project_favs.remove(detection_key)
                        action = "removed from favorites"
                        button_update = gr.update(value="Favorite", variant="secondary")
                    else:
                        project_favs.add(detection_key)
                        action = "added to favorites"
                        button_update = gr.update(value="Favorited", variant="primary")
                    updated_map[project_slug] = sorted(project_favs)
                    return f"Detection {detection_key} {action}", updated_map, button_update

                def update_favorite_button_state(
                    project_slug: str,
                    rows: object,
                    idx: int,
                    favorite_map: dict[str, list[str]],
                ):
                    normalized_rows = _normalize_rows(rows)
                    if not project_slug or not normalized_rows:
                        return gr.update(value="Favorite", variant="secondary")

                    safe_idx = max(0, min(int(idx), len(normalized_rows) - 1))
                    detection_key = str(normalized_rows[safe_idx][0]).strip()
                    favs = set((favorite_map or {}).get(project_slug, []))
                    if detection_key in favs:
                        return gr.update(value="Favorited", variant="primary")
                    return gr.update(value="Favorite", variant="secondary")

                def on_table_select(project_slug: str, repo: str, rows: object, cache_key: str, session, evt: gr.SelectData):
                    return _select_and_fetch_audio_with_title(
                        audio_service=audio_service,
                        dataset_repo=repo,
                        rows=rows,
                        cache_key=cache_key,
                        evt=evt,
                        allow_demo_fallback=False,
                        hf_token=_project_fetch_token(project_slug, session),
                    )

                refresh_event = refresh_btn.click(
                    fn=refresh,
                    inputs=[
                        selected_project_state,
                        page_state,
                        species_filter,
                        min_confidence,
                        validator_filter,
                        validation_status_filter,
                        updated_after_filter,
                        show_conflicts_only,
                    ],
                    outputs=[table, status, page_state],
                )
                next_event = next_btn.click(
                    fn=go_next,
                    inputs=[
                        selected_project_state,
                        page_state,
                        species_filter,
                        min_confidence,
                        validator_filter,
                        validation_status_filter,
                        updated_after_filter,
                        show_conflicts_only,
                    ],
                    outputs=[table, status, page_state],
                )
                prev_event = prev_btn.click(
                    fn=go_prev,
                    inputs=[
                        selected_project_state,
                        page_state,
                        species_filter,
                        min_confidence,
                        validator_filter,
                        validation_status_filter,
                        updated_after_filter,
                        show_conflicts_only,
                    ],
                    outputs=[table, status, page_state],
                )

                selected_dataset_repo_state.change(
                    fn=lambda repo_id: gr.update(value=repo_id),
                    inputs=[selected_dataset_repo_state],
                    outputs=[dataset_repo],
                )
                project_change_event = selected_project_state.change(
                    fn=refresh_for_selected_project,
                    inputs=[selected_project_state, session_state],
                    outputs=[species_filter, table, status, page_state, audio_player, spectrogram_image, spectrogram_title, validation_summary_cards, corrected_species_input, project_species_state],
                )

                species_filter.change(
                    fn=lambda project_slug, species, confidence, validator_filter_value, status_filter_value, updated_after_value, only_conflicts: refresh(
                        project_slug,
                        1,
                        species,
                        confidence,
                        validator_filter_value,
                        status_filter_value,
                        updated_after_value,
                        only_conflicts,
                    ),
                    inputs=[
                        selected_project_state,
                        species_filter,
                        min_confidence,
                        validator_filter,
                        validation_status_filter,
                        updated_after_filter,
                        show_conflicts_only,
                    ],
                    outputs=[table, status, page_state],
                ).then(
                    fn=lambda rows: _build_validation_summary_cards(rows),
                    inputs=[table],
                    outputs=[validation_summary_cards],
                ).then(
                    fn=lambda project_slug, repo, rows, cache_key, session: _autofetch_first_row_with_title(
                        audio_service=audio_service,
                        dataset_repo=repo,
                        rows=rows,
                        cache_key=cache_key,
                        allow_demo_fallback=False,
                        hf_token=_project_fetch_token(project_slug, session),
                    ),
                    inputs=[selected_project_state, selected_dataset_repo_state, table, cache_key_state, session_state],
                    outputs=[selected_index, audio_player, cache_key_state, status, spectrogram_image, spectrogram_title],
                ).then(
                    fn=lambda rows, idx: _mark_selected_row(rows, int(idx)),
                    inputs=[table, selected_index],
                    outputs=[table],
                ).then(
                    fn=update_favorite_button_state,
                    inputs=[selected_project_state, table, selected_index, favorite_detection_state],
                    outputs=[favorite_btn],
                )

                table_select_event = table.select(
                    fn=on_table_select,
                    inputs=[selected_project_state, selected_dataset_repo_state, table, cache_key_state, session_state],
                    outputs=[selected_index, audio_player, cache_key_state, status, spectrogram_image, spectrogram_title],
                )
                table_select_event.then(
                    fn=update_favorite_button_state,
                    inputs=[selected_project_state, table, selected_index, favorite_detection_state],
                    outputs=[favorite_btn],
                )

                selected_index.change(
                    fn=lambda rows, idx: _selected_segment_card(rows, int(idx)),
                    inputs=[table, selected_index],
                    outputs=[selected_segment_card],
                )
                table.change(
                    fn=lambda rows, idx: _selected_segment_card(rows, int(idx)),
                    inputs=[table, selected_index],
                    outputs=[selected_segment_card],
                )

                auto_play_audio.change(
                    fn=lambda enabled: gr.update(autoplay=bool(enabled)),
                    inputs=[auto_play_audio],
                    outputs=[audio_player],
                )

                refresh_event.then(
                    fn=lambda rows: _build_validation_summary_cards(rows),
                    inputs=[table],
                    outputs=[validation_summary_cards],
                ).then(
                    fn=lambda project_slug, repo, rows, cache_key, session: _autofetch_first_row_with_title(
                        audio_service=audio_service,
                        dataset_repo=repo,
                        rows=rows,
                        cache_key=cache_key,
                        allow_demo_fallback=False,
                        hf_token=_project_fetch_token(project_slug, session),
                    ),
                    inputs=[selected_project_state, selected_dataset_repo_state, table, cache_key_state, session_state],
                    outputs=[selected_index, audio_player, cache_key_state, status, spectrogram_image, spectrogram_title],
                ).then(
                    fn=lambda rows, idx: _mark_selected_row(rows, int(idx)),
                    inputs=[table, selected_index],
                    outputs=[table],
                ).then(
                    fn=update_favorite_button_state,
                    inputs=[selected_project_state, table, selected_index, favorite_detection_state],
                    outputs=[favorite_btn],
                )
                next_event.then(
                    fn=lambda rows: _build_validation_summary_cards(rows),
                    inputs=[table],
                    outputs=[validation_summary_cards],
                ).then(
                    fn=lambda project_slug, repo, rows, cache_key, session: _autofetch_first_row_with_title(
                        audio_service=audio_service,
                        dataset_repo=repo,
                        rows=rows,
                        cache_key=cache_key,
                        allow_demo_fallback=False,
                        hf_token=_project_fetch_token(project_slug, session),
                    ),
                    inputs=[selected_project_state, selected_dataset_repo_state, table, cache_key_state, session_state],
                    outputs=[selected_index, audio_player, cache_key_state, status, spectrogram_image, spectrogram_title],
                ).then(
                    fn=lambda rows, idx: _mark_selected_row(rows, int(idx)),
                    inputs=[table, selected_index],
                    outputs=[table],
                ).then(
                    fn=update_favorite_button_state,
                    inputs=[selected_project_state, table, selected_index, favorite_detection_state],
                    outputs=[favorite_btn],
                )
                prev_event.then(
                    fn=lambda rows: _build_validation_summary_cards(rows),
                    inputs=[table],
                    outputs=[validation_summary_cards],
                ).then(
                    fn=lambda project_slug, repo, rows, cache_key, session: _autofetch_first_row_with_title(
                        audio_service=audio_service,
                        dataset_repo=repo,
                        rows=rows,
                        cache_key=cache_key,
                        allow_demo_fallback=False,
                        hf_token=_project_fetch_token(project_slug, session),
                    ),
                    inputs=[selected_project_state, selected_dataset_repo_state, table, cache_key_state, session_state],
                    outputs=[selected_index, audio_player, cache_key_state, status, spectrogram_image, spectrogram_title],
                ).then(
                    fn=lambda rows, idx: _mark_selected_row(rows, int(idx)),
                    inputs=[table, selected_index],
                    outputs=[table],
                ).then(
                    fn=update_favorite_button_state,
                    inputs=[selected_project_state, table, selected_index, favorite_detection_state],
                    outputs=[favorite_btn],
                )

                favorite_btn.click(
                    fn=toggle_favorite_detection,
                    inputs=[selected_project_state, table, selected_index, favorite_detection_state],
                    outputs=[status, favorite_detection_state, favorite_btn],
                )

                corrected_species_input.change(
                    fn=save_corrected_species_option,
                    inputs=[selected_project_state, corrected_species_input, project_species_state, custom_corrected_species_state],
                    outputs=[corrected_species_input, custom_corrected_species_state],
                )

                approve_event = approve_btn.click(
                    fn=lambda project_slug, rows, idx, session, notes, corrected_species_value, cache_key, page, species, confidence, validator_filter_value, status_filter_value, updated_after_value, only_conflicts: save_for_project(
                        project_slug,
                        "positive",
                        rows,
                        idx,
                        session,
                        notes,
                        corrected_species_value,
                        cache_key,
                        page,
                        species,
                        confidence,
                        validator_filter_value,
                        status_filter_value,
                        updated_after_value,
                        only_conflicts,
                    ),
                    inputs=[
                        selected_project_state,
                        table,
                        selected_index,
                        session_state,
                        validation_notes,
                        corrected_species_input,
                        cache_key_state,
                        page_state,
                        species_filter,
                        min_confidence,
                        validator_filter,
                        validation_status_filter,
                        updated_after_filter,
                        show_conflicts_only,
                    ],
                    outputs=[status, cache_key_state, audio_player, table, page_state, selected_index, pending_status_state, conflict_detection_key_state],
                )
                reject_event = reject_btn.click(
                    fn=lambda project_slug, rows, idx, session, notes, corrected_species_value, cache_key, page, species, confidence, validator_filter_value, status_filter_value, updated_after_value, only_conflicts: save_for_project(
                        project_slug,
                        "negative",
                        rows,
                        idx,
                        session,
                        notes,
                        corrected_species_value,
                        cache_key,
                        page,
                        species,
                        confidence,
                        validator_filter_value,
                        status_filter_value,
                        updated_after_value,
                        only_conflicts,
                    ),
                    inputs=[
                        selected_project_state,
                        table,
                        selected_index,
                        session_state,
                        validation_notes,
                        corrected_species_input,
                        cache_key_state,
                        page_state,
                        species_filter,
                        min_confidence,
                        validator_filter,
                        validation_status_filter,
                        updated_after_filter,
                        show_conflicts_only,
                    ],
                    outputs=[status, cache_key_state, audio_player, table, page_state, selected_index, pending_status_state, conflict_detection_key_state],
                )
                uncertain_event = uncertain_btn.click(
                    fn=lambda project_slug, rows, idx, session, notes, corrected_species_value, cache_key, page, species, confidence, validator_filter_value, status_filter_value, updated_after_value, only_conflicts: save_for_project(
                        project_slug,
                        "uncertain",
                        rows,
                        idx,
                        session,
                        notes,
                        corrected_species_value,
                        cache_key,
                        page,
                        species,
                        confidence,
                        validator_filter_value,
                        status_filter_value,
                        updated_after_value,
                        only_conflicts,
                    ),
                    inputs=[
                        selected_project_state,
                        table,
                        selected_index,
                        session_state,
                        validation_notes,
                        corrected_species_input,
                        cache_key_state,
                        page_state,
                        species_filter,
                        min_confidence,
                        validator_filter,
                        validation_status_filter,
                        updated_after_filter,
                        show_conflicts_only,
                    ],
                    outputs=[status, cache_key_state, audio_player, table, page_state, selected_index, pending_status_state, conflict_detection_key_state],
                )
                skip_event = skip_btn.click(
                    fn=lambda project_slug, rows, idx, session, notes, corrected_species_value, cache_key, page, species, confidence, validator_filter_value, status_filter_value, updated_after_value, only_conflicts: save_for_project(
                        project_slug,
                        "skip",
                        rows,
                        idx,
                        session,
                        notes,
                        corrected_species_value,
                        cache_key,
                        page,
                        species,
                        confidence,
                        validator_filter_value,
                        status_filter_value,
                        updated_after_value,
                        only_conflicts,
                    ),
                    inputs=[
                        selected_project_state,
                        table,
                        selected_index,
                        session_state,
                        validation_notes,
                        corrected_species_input,
                        cache_key_state,
                        page_state,
                        species_filter,
                        min_confidence,
                        validator_filter,
                        validation_status_filter,
                        updated_after_filter,
                        show_conflicts_only,
                    ],
                    outputs=[status, cache_key_state, audio_player, table, page_state, selected_index, pending_status_state, conflict_detection_key_state],
                )

                approve_event.then(
                    fn=lambda project_slug, repo, rows, idx, cache_key, session: _advance_to_next_row_with_title(
                        audio_service=audio_service,
                        dataset_repo=repo,
                        rows=rows,
                        selected_index=int(idx),
                        cache_key=cache_key,
                        allow_demo_fallback=False,
                        hf_token=_project_fetch_token(project_slug, session),
                    ),
                    inputs=[selected_project_state, selected_dataset_repo_state, table, selected_index, cache_key_state, session_state],
                    outputs=[selected_index, audio_player, cache_key_state, status, spectrogram_image, spectrogram_title],
                ).then(
                    fn=lambda rows, idx: _mark_selected_row(rows, int(idx)),
                    inputs=[table, selected_index],
                    outputs=[table],
                ).then(
                    fn=update_favorite_button_state,
                    inputs=[selected_project_state, table, selected_index, favorite_detection_state],
                    outputs=[favorite_btn],
                ).then(fn=lambda rows: _build_validation_summary_cards(rows), inputs=[table], outputs=[validation_summary_cards])

                reject_event.then(
                    fn=lambda project_slug, repo, rows, idx, cache_key, session: _advance_to_next_row_with_title(
                        audio_service=audio_service,
                        dataset_repo=repo,
                        rows=rows,
                        selected_index=int(idx),
                        cache_key=cache_key,
                        allow_demo_fallback=False,
                        hf_token=_project_fetch_token(project_slug, session),
                    ),
                    inputs=[selected_project_state, selected_dataset_repo_state, table, selected_index, cache_key_state, session_state],
                    outputs=[selected_index, audio_player, cache_key_state, status, spectrogram_image, spectrogram_title],
                ).then(
                    fn=lambda rows, idx: _mark_selected_row(rows, int(idx)),
                    inputs=[table, selected_index],
                    outputs=[table],
                ).then(
                    fn=update_favorite_button_state,
                    inputs=[selected_project_state, table, selected_index, favorite_detection_state],
                    outputs=[favorite_btn],
                ).then(fn=lambda rows: _build_validation_summary_cards(rows), inputs=[table], outputs=[validation_summary_cards])

                uncertain_event.then(
                    fn=lambda project_slug, repo, rows, idx, cache_key, session: _advance_to_next_row_with_title(
                        audio_service=audio_service,
                        dataset_repo=repo,
                        rows=rows,
                        selected_index=int(idx),
                        cache_key=cache_key,
                        allow_demo_fallback=False,
                        hf_token=_project_fetch_token(project_slug, session),
                    ),
                    inputs=[selected_project_state, selected_dataset_repo_state, table, selected_index, cache_key_state, session_state],
                    outputs=[selected_index, audio_player, cache_key_state, status, spectrogram_image, spectrogram_title],
                ).then(
                    fn=lambda rows, idx: _mark_selected_row(rows, int(idx)),
                    inputs=[table, selected_index],
                    outputs=[table],
                ).then(
                    fn=update_favorite_button_state,
                    inputs=[selected_project_state, table, selected_index, favorite_detection_state],
                    outputs=[favorite_btn],
                ).then(fn=lambda rows: _build_validation_summary_cards(rows), inputs=[table], outputs=[validation_summary_cards])

                skip_event.then(
                    fn=lambda project_slug, repo, rows, idx, cache_key, session: _advance_to_next_row_with_title(
                        audio_service=audio_service,
                        dataset_repo=repo,
                        rows=rows,
                        selected_index=int(idx),
                        cache_key=cache_key,
                        allow_demo_fallback=False,
                        hf_token=_project_fetch_token(project_slug, session),
                    ),
                    inputs=[selected_project_state, selected_dataset_repo_state, table, selected_index, cache_key_state, session_state],
                    outputs=[selected_index, audio_player, cache_key_state, status, spectrogram_image, spectrogram_title],
                ).then(
                    fn=lambda rows, idx: _mark_selected_row(rows, int(idx)),
                    inputs=[table, selected_index],
                    outputs=[table],
                ).then(
                    fn=update_favorite_button_state,
                    inputs=[selected_project_state, table, selected_index, favorite_detection_state],
                    outputs=[favorite_btn],
                ).then(fn=lambda rows: _build_validation_summary_cards(rows), inputs=[table], outputs=[validation_summary_cards])

                session_state.change(
                    fn=lambda s: gr.update(value=(s.username if s is not None else "")),
                    inputs=[session_state],
                    outputs=[validator_name],
                )

            # ===== TAB 5: Report =====
            with gr.Tab("Progress", id="report_tab"):
                gr.HTML(
                    section_header_html(
                        "Progress dashboard",
                        "Project validation health",
                        "Track coverage by status, species, and team activity for the selected project.",
                        class_name="bn-report-panel",
                    )
                )
                report_project_selector = gr.Dropdown(
                    choices=[],
                    value=None,
                    label="Project",
                    interactive=False,
                    allow_custom_value=True,
                )
                refresh_report_btn = gr.Button(
                    "Refresh dashboard",
                    variant="primary",
                    elem_classes=["bn-clean-action", "bn-orange-action"],
                )
                report_kpis = gr.HTML(value="")
                report_coverage_bars = gr.HTML(value=coverage_bars_html([]))
                report_validator_page = gr.State(value=1)
                report_recent_page = gr.State(value=1)
                report_validator_table = gr.HTML(
                    value=paged_activity_html("Validator activity", ["Validator", "Validations"], []),
                )
                with gr.Row(elem_classes=["bn-clean-button-row"]):
                    report_validator_prev_btn = gr.Button("Previous validator page", elem_classes=["bn-clean-action"])
                    report_validator_next_btn = gr.Button("Next validator page", elem_classes=["bn-clean-action"])
                report_recent_table = gr.HTML(
                    value=paged_activity_html("Recent activity", ["Timestamp", "Validator", "Status", "Detection"], []),
                )
                with gr.Row(elem_classes=["bn-clean-button-row"]):
                    report_recent_prev_btn = gr.Button("Previous recent page", elem_classes=["bn-clean-action"])
                    report_recent_next_btn = gr.Button("Next recent page", elem_classes=["bn-clean-action"])
                report_status = gr.Markdown("")
                with gr.Column(elem_classes=["bn-report-download-section"]):
                    with gr.Group(elem_classes=["bn-report-panel", "bn-report-download-panel"]):
                        gr.HTML(
                            section_header_html(
                                "Download",
                                "Complete validation dataset",
                                "Export every detection with the validation fields already filled in for the selected project.",
                                class_name="bn-panel-soft",
                            )
                        )
                        report_export_status = gr.Markdown("")
                        report_export_csv_file = gr.DownloadButton(
                            "Download prepared CSV",
                            elem_id="bn-report-export-csv-download",
                            elem_classes=["bn-autodownload-target"],
                        )
                        report_export_xlsx_file = gr.DownloadButton(
                            "Download prepared XLSX",
                            elem_id="bn-report-export-xlsx-download",
                            elem_classes=["bn-autodownload-target"],
                        )
                    with gr.Row(elem_classes=["bn-report-download-action-row"]):
                        report_export_csv_btn = gr.Button(
                            "Download CSV",
                            elem_classes=["bn-report-download-action", "bn-report-download-action-orange"],
                        )
                        report_export_xlsx_btn = gr.Button(
                            "Download XLSX",
                            elem_classes=["bn-report-download-action", "bn-report-download-action-blue"],
                        )

                def _list_project_detections(project_slug: str, session=None) -> list[Detection]:
                    if not project_slug:
                        return []
                    _ensure_project_queue_loaded(project_slug, session)
                    return service_ref["queue"].list_all_detections(project_slug=project_slug)

                def _prepare_report_export(project_slug: str, file_format: str, session):
                    slug = (project_slug or "").strip()
                    if session is None:
                        return None, "Login before exporting project data."
                    if not slug or slug not in session.authorized_projects:
                        return None, "Choose an authorized project before exporting."

                    project = _project_map().get(slug)
                    items = _list_project_detections(slug, session)
                    if not items:
                        return None, f"Project '{slug}' has no detections loaded for export."

                    snapshot = validation_repository.load_current_snapshot(project_slug=slug)
                    output_path = _write_validation_export(
                        items,
                        snapshot,
                        project_slug=slug,
                        dataset_repo_id=(project.dataset_repo_id if project is not None else ""),
                        file_format=file_format,
                    )
                    reviewed = sum(
                        1
                        for detection in items
                        if str(snapshot.get(detection.detection_key, {}).get("status") or "pending").lower() != "pending"
                    )
                    return (
                        str(output_path),
                        (
                            f"Prepared **{file_format.upper()}** with **{len(items)}** detections and "
                            f"**{reviewed}** current validations for project **{slug}**."
                        ),
                    )

                def _render_report_dashboard(project_slug: str, validator_page: int = 1, recent_page: int = 1, session=None):
                    slug = (project_slug or "").strip()
                    try:
                        validator_page = max(1, int(validator_page or 1))
                    except (TypeError, ValueError):
                        validator_page = 1
                    try:
                        recent_page = max(1, int(recent_page or 1))
                    except (TypeError, ValueError):
                        recent_page = 1
                    if not slug:
                        return (
                            "",
                            coverage_bars_html([]),
                            paged_activity_html("Validator activity", ["Validator", "Validations"], []),
                            paged_activity_html("Recent activity", ["Timestamp", "Validator", "Status", "Detection"], []),
                            1,
                            1,
                            "Select a project to view the dashboard",
                        )

                    warning = _ensure_project_queue_loaded(slug, session)
                    signature = loaded_project_signatures.get(slug, "")
                    cache_key = (slug, validator_page, recent_page, signature)
                    cached = report_cache.get(cache_key)
                    if cached and time.monotonic() - cached[0] <= report_cache_ttl_seconds:
                        return cached[1]

                    items = service_ref["queue"].list_all_detections(project_slug=slug)
                    snapshot = validation_repository.load_current_snapshot(project_slug=slug)
                    events = validation_repository.list_events(project_slug=slug)
                    total_recordings = len(items)

                    species_totals: dict[str, dict[str, int]] = {}
                    status_totals = {"positive": 0, "negative": 0, "uncertain": 0, "skip": 0}
                    validator_totals: dict[str, int] = {}
                    validated_recordings = 0

                    for item in items:
                        species_name = str(item.scientific_name).strip() or "Unknown species"
                        counters = species_totals.setdefault(species_name, {"total": 0, "validated": 0})
                        counters["total"] += 1

                        state = snapshot.get(item.detection_key, {})
                        status_value = str(state.get("status", "pending")).strip().lower()
                        if status_value and status_value != "pending":
                            counters["validated"] += 1
                            validated_recordings += 1
                            if status_value in status_totals:
                                status_totals[status_value] += 1
                            validator = str(state.get("validator", "")).strip() or "unknown"
                            validator_totals[validator] = validator_totals.get(validator, 0) + 1

                    validated_species = sum(1 for counters in species_totals.values() if counters["validated"] > 0)
                    remaining_recordings = max(0, total_recordings - validated_recordings)
                    coverage_pct = round((validated_recordings / total_recordings) * 100, 1) if total_recordings else 0.0

                    rows = []
                    for species_name, counters in species_totals.items():
                        remaining = max(0, counters["total"] - counters["validated"])
                        species_coverage = round((counters["validated"] / counters["total"]) * 100, 1) if counters["total"] else 0.0
                        rows.append([species_name, counters["total"], counters["validated"], remaining, species_coverage])
                    rows.sort(key=lambda row: (-int(row[1]), str(row[0]).lower()))

                    validator_rows = sorted(
                        [[validator, total] for validator, total in validator_totals.items()],
                        key=lambda row: (-int(row[1]), str(row[0]).lower()),
                    )
                    recent_rows = []
                    for event in sorted(events, key=lambda payload: str(payload.get("timestamp") or payload.get("created_at") or ""), reverse=True):
                        recent_rows.append(
                            [
                                str(event.get("timestamp") or event.get("created_at") or ""),
                                str(event.get("validator") or ""),
                                str(event.get("status") or ""),
                                str(event.get("detection_key") or ""),
                            ]
                        )

                    def clamp_activity_page(activity_rows: list[list[object]], requested_page: int) -> int:
                        total_pages = max(1, ((len(activity_rows) - 1) // 10) + 1) if activity_rows else 1
                        return max(1, min(int(requested_page), total_pages))

                    validator_page = clamp_activity_page(validator_rows, validator_page)
                    recent_page = clamp_activity_page(recent_rows, recent_page)

                    kpis_html = compact_metric_grid(
                        [
                            ("Coverage", f"{coverage_pct}%", f"{validated_recordings} of {total_recordings}", "info"),
                            ("Remaining", str(remaining_recordings), "segments pending", "warning"),
                            ("Confirmed", str(status_totals["positive"]), "accepted segments", "positive"),
                            ("Rejected", str(status_totals["negative"]), "negative segments", "negative"),
                            ("Uncertain", str(status_totals["uncertain"]), "needs review", ""),
                            ("Skipped", str(status_totals["skip"]), "not reviewed", ""),
                            ("Species touched", str(validated_species), "with validation", ""),
                            ("Validators", str(len(validator_totals)), "active in project", ""),
                        ]
                    )
                    status_text = (
                        f"Project: **{slug}** | Total recordings: **{total_recordings}** | "
                        f"Validated: **{validated_recordings}** | Remaining: **{remaining_recordings}**"
                    )
                    if warning:
                        status_text = f"{status_text}\n\n{warning}"
                    result = (
                        kpis_html,
                        coverage_bars_html(rows),
                        paged_activity_html(
                            "Validator activity",
                            ["Validator", "Validations"],
                            validator_rows,
                            page=validator_page,
                        ),
                        paged_activity_html(
                            "Recent activity",
                            ["Timestamp", "Validator", "Status", "Detection"],
                            recent_rows,
                            page=recent_page,
                        ),
                        validator_page,
                        recent_page,
                        status_text,
                    )
                    report_cache[cache_key] = (time.monotonic(), result)
                    return result

                session_state.change(
                    fn=lambda s: (
                        gr.update(
                            choices=(s.authorized_projects if s is not None else []),
                            value=(s.authorized_projects[0] if (s is not None and s.authorized_projects) else None),
                            interactive=bool(s is not None and s.authorized_projects),
                        ),
                        "",
                        coverage_bars_html([]),
                        paged_activity_html("Validator activity", ["Validator", "Validations"], []),
                        paged_activity_html("Recent activity", ["Timestamp", "Validator", "Status", "Detection"], []),
                        1,
                        1,
                        "Login and choose a project." if s is None else "Choose a project to load project metrics.",
                    ),
                    inputs=[session_state],
                    outputs=[report_project_selector, report_kpis, report_coverage_bars, report_validator_table, report_recent_table, report_validator_page, report_recent_page, report_status],
                )

                report_project_selector.change(
                    fn=lambda project_slug, session: _render_report_dashboard(project_slug, 1, 1, session),
                    inputs=[report_project_selector, session_state],
                    outputs=[report_kpis, report_coverage_bars, report_validator_table, report_recent_table, report_validator_page, report_recent_page, report_status],
                )

                selected_project_state.change(
                    fn=lambda p: gr.update(value=p if p else None),
                    inputs=[selected_project_state],
                    outputs=[report_project_selector],
                ).then(
                    fn=lambda project_slug, session: _render_report_dashboard(project_slug, 1, 1, session),
                    inputs=[selected_project_state, session_state],
                    outputs=[report_kpis, report_coverage_bars, report_validator_table, report_recent_table, report_validator_page, report_recent_page, report_status],
                )

                refresh_report_btn.click(
                    fn=_render_report_dashboard,
                    inputs=[report_project_selector, report_validator_page, report_recent_page, session_state],
                    outputs=[report_kpis, report_coverage_bars, report_validator_table, report_recent_table, report_validator_page, report_recent_page, report_status],
                )

                report_validator_prev_btn.click(
                    fn=lambda project_slug, page, recent, session: _render_report_dashboard(project_slug, max(1, int(page) - 1), int(recent), session),
                    inputs=[report_project_selector, report_validator_page, report_recent_page, session_state],
                    outputs=[report_kpis, report_coverage_bars, report_validator_table, report_recent_table, report_validator_page, report_recent_page, report_status],
                )
                report_validator_next_btn.click(
                    fn=lambda project_slug, page, recent, session: _render_report_dashboard(project_slug, int(page) + 1, int(recent), session),
                    inputs=[report_project_selector, report_validator_page, report_recent_page, session_state],
                    outputs=[report_kpis, report_coverage_bars, report_validator_table, report_recent_table, report_validator_page, report_recent_page, report_status],
                )
                report_recent_prev_btn.click(
                    fn=lambda project_slug, validator, page, session: _render_report_dashboard(project_slug, int(validator), max(1, int(page) - 1), session),
                    inputs=[report_project_selector, report_validator_page, report_recent_page, session_state],
                    outputs=[report_kpis, report_coverage_bars, report_validator_table, report_recent_table, report_validator_page, report_recent_page, report_status],
                )
                report_recent_next_btn.click(
                    fn=lambda project_slug, validator, page, session: _render_report_dashboard(project_slug, int(validator), int(page) + 1, session),
                    inputs=[report_project_selector, report_validator_page, report_recent_page, session_state],
                    outputs=[report_kpis, report_coverage_bars, report_validator_table, report_recent_table, report_validator_page, report_recent_page, report_status],
                )
                report_export_csv_event = report_export_csv_btn.click(
                    fn=lambda project_slug, session: _prepare_report_export(project_slug, "csv", session),
                    inputs=[report_project_selector, session_state],
                    outputs=[report_export_csv_file, report_export_status],
                )
                report_export_csv_event.then(
                    fn=None,
                    js="() => document.querySelector('#bn-report-export-csv-download button, #bn-report-export-csv-download')?.click()",
                )
                report_export_xlsx_event = report_export_xlsx_btn.click(
                    fn=lambda project_slug, session: _prepare_report_export(project_slug, "xlsx", session),
                    inputs=[report_project_selector, session_state],
                    outputs=[report_export_xlsx_file, report_export_status],
                )
                report_export_xlsx_event.then(
                    fn=None,
                    js="() => document.querySelector('#bn-report-export-xlsx-download button, #bn-report-export-xlsx-download')?.click()",
                )

            # ===== TAB 6: Settings =====
            with gr.Tab("Settings", id="settings_tab"):
                gr.HTML(
                    section_header_html(
                        "System health",
                        "Runtime configuration and deployment checks",
                        "Use this page to confirm the active backend, storage mode, and operational settings without exposing secrets.",
                    )
                )
                settings_health = gr.HTML(value="")
                settings_status = gr.Markdown(value="")
                refresh_settings_btn = gr.Button(
                    "Refresh health",
                    elem_classes=["bn-orange-action"],
                )

                def _render_settings_health():
                    backend = (runtime_config.state_backend or "filesystem").strip().lower()
                    supabase_ready = bool(runtime_config.supabase_url and runtime_config.supabase_service_role_key)
                    state_label = "Supabase" if backend in {"supabase", "postgres", "postgresql"} and supabase_ready else "Filesystem"
                    state_tone = "ok" if state_label == "Supabase" else "warn"
                    demo_label = "enabled" if runtime_config.enable_demo_bootstrap else "disabled"
                    demo_tone = "warn" if runtime_config.enable_demo_bootstrap else "ok"
                    invite_email_label = "enabled" if runtime_config.invite_email_enabled and runtime_config.emailjs_enabled else "disabled"
                    hf_space_label = os.getenv("SPACE_ID") or "local runtime"
                    health_html = settings_health_html(
                        [
                            ("State backend", state_label, state_tone),
                            ("Supabase URL", "configured" if runtime_config.supabase_url else "missing", "ok" if runtime_config.supabase_url else "warn"),
                            ("Supabase service role", "configured" if runtime_config.supabase_service_role_key else "missing", "ok" if runtime_config.supabase_service_role_key else "warn"),
                            ("Demo bootstrap", demo_label, demo_tone),
                            ("Invite email", invite_email_label, "ok" if invite_email_label == "enabled" else ""),
                            ("Page size", str(runtime_config.page_size), ""),
                            ("Validation storage", runtime_config.validation_base_dir, ""),
                            ("Runtime", hf_space_label, "info"),
                        ]
                    )
                    status_text = (
                        "Supabase persistence is active."
                        if state_label == "Supabase"
                        else "Filesystem persistence is active. On free Spaces, local files are not durable across rebuilds."
                    )
                    return health_html, status_text

                wrapper.load(
                    fn=_render_settings_health,
                    outputs=[settings_health, settings_status],
                )
                refresh_settings_btn.click(
                    fn=_render_settings_health,
                    outputs=[settings_health, settings_status],
                )

    return wrapper
