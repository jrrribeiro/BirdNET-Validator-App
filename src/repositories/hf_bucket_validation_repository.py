import json
import tempfile
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from huggingface_hub import batch_bucket_files, create_bucket, download_bucket_files, list_bucket_tree

from src.domain.models import Validation
from src.repositories.append_only_validation_repository import OptimisticLockError


HF_BUCKET_VALIDATION_BACKEND = "hf_bucket"
HF_BUCKET_VALIDATION_SCHEMA_VERSION = 1
HF_BUCKET_DEFAULT_ACTIVE_EVENT_LIMIT = 250


class HfBucketValidationError(Exception):
    """Raised when validation state cannot be read from or written to an HF Bucket."""


def default_validation_bucket_id(dataset_repo_id: str) -> str:
    repo_id = (dataset_repo_id or "").strip().strip("/")
    if "/" not in repo_id:
        raise HfBucketValidationError("Dataset repo id must be in owner/name format to create a validation bucket.")
    namespace, name = repo_id.split("/", 1)
    if not namespace.strip() or not name.strip():
        raise HfBucketValidationError("Dataset repo id must be in owner/name format to create a validation bucket.")
    suffix = name if name.endswith("_validation_state") else f"{name}_validation_state"
    return f"{namespace}/{suffix}"


class HfBucketFilesApi(Protocol):
    def create_private_bucket(self, *, bucket_id: str, token: str) -> None: ...

    def list_files(self, *, bucket_id: str, prefix: str, token: str) -> list[str]: ...

    def read_text(self, *, bucket_id: str, path_in_bucket: str, token: str) -> str: ...

    def read_texts(self, *, bucket_id: str, paths_in_bucket: list[str], token: str) -> dict[str, str]: ...

    def write_files(
        self,
        *,
        bucket_id: str,
        files: dict[str, bytes],
        token: str,
        delete_paths: list[str] | None = None,
    ) -> None: ...


class HuggingFaceBucketFilesApi:
    def create_private_bucket(self, *, bucket_id: str, token: str) -> None:
        create_bucket(bucket_id, private=True, exist_ok=True, token=token)

    def list_files(self, *, bucket_id: str, prefix: str, token: str) -> list[str]:
        return [
            item.path
            for item in list_bucket_tree(bucket_id, prefix=prefix, recursive=True, token=token)
            if getattr(item, "type", "") == "file"
        ]

    def read_text(self, *, bucket_id: str, path_in_bucket: str, token: str) -> str:
        downloaded = self.read_texts(
            bucket_id=bucket_id,
            paths_in_bucket=[path_in_bucket],
            token=token,
        )
        if path_in_bucket not in downloaded:
            raise FileNotFoundError(path_in_bucket)
        return downloaded[path_in_bucket]

    def read_texts(self, *, bucket_id: str, paths_in_bucket: list[str], token: str) -> dict[str, str]:
        if not paths_in_bucket:
            return {}
        with tempfile.TemporaryDirectory(prefix="birdnet-bucket-read-") as temp_dir:
            downloads: list[tuple[str, Path]] = []
            for path_in_bucket in paths_in_bucket:
                local_path = Path(temp_dir) / Path(path_in_bucket)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                downloads.append((path_in_bucket, local_path))
            download_bucket_files(
                bucket_id,
                downloads,
                raise_on_missing_files=False,
                token=token,
            )
            return {
                path_in_bucket: local_path.read_text(encoding="utf-8")
                for path_in_bucket, local_path in downloads
                if local_path.exists()
            }

    def write_files(
        self,
        *,
        bucket_id: str,
        files: dict[str, bytes],
        token: str,
        delete_paths: list[str] | None = None,
    ) -> None:
        batch_bucket_files(
            bucket_id,
            add=[(contents, path) for path, contents in files.items()],
            delete=delete_paths or None,
            token=token,
        )


@dataclass(frozen=True)
class HfBucketValidationInitResult:
    bucket_id: str
    initialized: bool
    reused_existing: bool


@dataclass(frozen=True)
class HfBucketCompactionResult:
    archive_path: str | None
    compacted_event_count: int


class HfBucketValidationInitializer:
    """Create a private mutable validation store owned by the project admin."""

    def __init__(self, api: HfBucketFilesApi | None = None) -> None:
        self._api = api or HuggingFaceBucketFilesApi()

    def initialize(self, *, project_slug: str, dataset_repo_id: str, token: str | None) -> HfBucketValidationInitResult:
        token_value = (token or "").strip()
        if not token_value:
            raise HfBucketValidationError("A Hugging Face token is required to create the private validation bucket.")
        bucket_id = default_validation_bucket_id(dataset_repo_id)
        try:
            self._api.create_private_bucket(bucket_id=bucket_id, token=token_value)
            existing_files = set(self._api.list_files(bucket_id=bucket_id, prefix="", token=token_value))
            if "metadata/project.json" in existing_files:
                return HfBucketValidationInitResult(bucket_id=bucket_id, initialized=False, reused_existing=True)
            if existing_files:
                sample = ", ".join(sorted(existing_files)[:5])
                raise HfBucketValidationError(
                    "The validation bucket contains files but no BirdNET manifest; "
                    f"refusing automatic initialization. Existing files: {sample}"
                )
            payload = {
                "schema_version": HF_BUCKET_VALIDATION_SCHEMA_VERSION,
                "project_slug": project_slug,
                "dataset_repo_id": dataset_repo_id,
                "validation_backend": HF_BUCKET_VALIDATION_BACKEND,
                "bucket_id": bucket_id,
                "created_at": datetime.now(UTC).isoformat(),
            }
            snapshot = {
                "schema_version": HF_BUCKET_VALIDATION_SCHEMA_VERSION,
                "project_slug": project_slug,
                "items": {},
            }
            self._api.write_files(
                bucket_id=bucket_id,
                token=token_value,
                files={
                    "metadata/project.json": _json_bytes(payload),
                    "snapshots/current.json": _json_bytes(snapshot),
                },
            )
        except HfBucketValidationError:
            raise
        except Exception as exc:
            raise HfBucketValidationError(f"Could not initialize private HF validation bucket {bucket_id}: {exc}") from exc
        return HfBucketValidationInitResult(bucket_id=bucket_id, initialized=True, reused_existing=False)


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")


def _extract_items(payload: object, project_slug: str) -> dict[str, dict[str, object]] | None:
    if not isinstance(payload, dict):
        return None
    payload_project = str(payload.get("project_slug") or project_slug).strip()
    if payload_project and project_slug and payload_project != project_slug:
        return None
    raw_items = payload.get("items", payload)
    if not isinstance(raw_items, dict):
        return None
    return {
        str(key): dict(value)
        for key, value in raw_items.items()
        if str(key).strip() and isinstance(value, dict)
    }


class HfBucketValidationRepository:
    """Mutable validation state in an admin-owned HF Bucket, without Git commits."""

    def __init__(
        self,
        *,
        bucket_id: str,
        token: str,
        api: HfBucketFilesApi | None = None,
        active_event_limit: int = HF_BUCKET_DEFAULT_ACTIVE_EVENT_LIMIT,
    ) -> None:
        self._bucket_id = (bucket_id or "").strip()
        self._token = (token or "").strip()
        self._api = api or HuggingFaceBucketFilesApi()
        self._active_event_limit = max(1, int(active_event_limit))
        self._lock = threading.RLock()
        if not self._bucket_id:
            raise HfBucketValidationError("A Hugging Face Bucket id is required.")
        if not self._token:
            raise HfBucketValidationError("A Hugging Face token is required to access validation state.")

    def _read_json_or_none(self, path_in_bucket: str) -> object | None:
        try:
            text = self._api.read_text(
                bucket_id=self._bucket_id,
                path_in_bucket=path_in_bucket,
                token=self._token,
            )
        except FileNotFoundError:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _active_event_paths(self) -> list[str]:
        return [
            path
            for path in self._api.list_files(bucket_id=self._bucket_id, prefix="events/", token=self._token)
            if path.endswith(".json")
        ]

    def _archive_paths(self) -> list[str]:
        return [
            path
            for path in self._api.list_files(bucket_id=self._bucket_id, prefix="archives/events/", token=self._token)
            if path.endswith(".jsonl")
        ]

    def _read_event_files(self, paths: list[str], *, project_slug: str) -> list[dict[str, object]]:
        texts = self._api.read_texts(bucket_id=self._bucket_id, paths_in_bucket=paths, token=self._token)
        events: list[dict[str, object]] = []
        for path in sorted(paths):
            text = texts.get(path)
            if text is None:
                continue
            payloads = text.splitlines() if path.endswith(".jsonl") else [text]
            for raw_payload in payloads:
                if not raw_payload.strip():
                    continue
                try:
                    payload = json.loads(raw_payload)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if str(payload.get("project_slug") or project_slug).strip() != project_slug:
                    continue
                events.append(payload)
        ordered = sorted(
            events,
            key=lambda event: (str(event.get("timestamp") or ""), str(event.get("event_id") or "")),
        )
        unique_events: list[dict[str, object]] = []
        seen_ids: set[str] = set()
        for event in ordered:
            event_id = str(event.get("event_id") or "").strip()
            if event_id and event_id in seen_ids:
                continue
            if event_id:
                seen_ids.add(event_id)
            unique_events.append(event)
        return unique_events

    def _active_events(self, project_slug: str) -> list[dict[str, object]]:
        return self._read_event_files(self._active_event_paths(), project_slug=project_slug)

    def list_events(self, project_slug: str, actor_username: str = "") -> list[dict[str, object]]:
        _ = actor_username
        return self._read_event_files(
            [*self._archive_paths(), *self._active_event_paths()],
            project_slug=project_slug,
        )

    def list_recent_events(
        self,
        project_slug: str,
        *,
        limit: int = 10,
        actor_username: str = "",
    ) -> list[dict[str, object]]:
        """Read only enough newest archives to serve recent-activity pages."""
        _ = actor_username
        requested = max(1, int(limit))
        events = self._active_events(project_slug)
        for index, archive_path in enumerate(sorted(self._archive_paths(), reverse=True)):
            if index > 0 and len(events) >= requested:
                break
            events.extend(self._read_event_files([archive_path], project_slug=project_slug))
        return sorted(
            events,
            key=lambda event: (str(event.get("timestamp") or ""), str(event.get("event_id") or "")),
            reverse=True,
        )[:requested]

    @staticmethod
    def _state_from_event(event: dict[str, object]) -> dict[str, object]:
        return {
            "status": event.get("status"),
            "corrected_species": event.get("corrected_species"),
            "notes": event.get("notes") or "",
            "validator": event.get("validator"),
            "updated_at": event.get("timestamp"),
            "version": int(event.get("new_version") or 0),
        }

    @classmethod
    def _merge_snapshot_and_events(
        cls,
        snapshot: dict[str, dict[str, object]],
        events: list[dict[str, object]],
    ) -> dict[str, dict[str, object]]:
        merged = {key: dict(value) for key, value in snapshot.items()}
        for event in events:
            key = str(event.get("detection_key") or "").strip()
            if not key:
                continue
            incoming = cls._state_from_event(event)
            current = merged.get(key)
            if current is None:
                merged[key] = incoming
                continue
            incoming_version = int(incoming.get("version") or 0)
            current_version = int(current.get("version") or 0)
            if incoming_version > current_version:
                merged[key] = incoming
                continue
            if incoming_version != current_version or incoming_version <= 0:
                continue
            state_fields = ("status", "corrected_species", "notes", "validator")
            incoming_decision = tuple(incoming.get(field) for field in state_fields)
            current_decision = tuple(current.get(field) for field in state_fields)
            incoming_is_newer = str(incoming.get("updated_at") or "") > str(current.get("updated_at") or "")
            if incoming_decision != current_decision:
                winner = dict(incoming if incoming_is_newer else current)
                winner["conflict"] = True
                winner["conflict_reason"] = "parallel_events_same_version"
                merged[key] = winner
                continue
            if incoming_is_newer and not bool(current.get("conflict")):
                merged[key] = incoming
        return merged

    def _snapshot_from_events(self, project_slug: str) -> dict[str, dict[str, object]]:
        return self._merge_snapshot_and_events({}, self.list_events(project_slug))

    def load_current_snapshot(self, project_slug: str, actor_username: str = "") -> dict[str, dict[str, object]]:
        _ = actor_username
        with self._lock:
            payload = self._read_json_or_none("snapshots/current.json")
            items = _extract_items(payload, project_slug)
            if items is None:
                return self._snapshot_from_events(project_slug)
            return self._merge_snapshot_and_events(items, self._active_events(project_slug))

    def compact_events(self, project_slug: str, *, force: bool = False) -> HfBucketCompactionResult:
        """Roll active event objects into an audit archive while preserving recoverability."""
        project = (project_slug or "").strip()
        if not project:
            raise HfBucketValidationError("A project slug is required to compact validation state.")

        with self._lock:
            active_paths = self._active_event_paths()
            if not force and len(active_paths) < self._active_event_limit:
                return HfBucketCompactionResult(archive_path=None, compacted_event_count=0)
            if not active_paths:
                return HfBucketCompactionResult(archive_path=None, compacted_event_count=0)

            events = self._read_event_files(active_paths, project_slug=project)
            if len(events) != len(active_paths):
                raise HfBucketValidationError(
                    "Could not compact active validation events safely because one or more files are unreadable."
                )
            snapshot_payload = self._read_json_or_none("snapshots/current.json")
            snapshot_items = _extract_items(snapshot_payload, project)
            if snapshot_items is None:
                snapshot_items = self._snapshot_from_events(project)
            reconciled_items = self._merge_snapshot_and_events(snapshot_items, events)
            now = datetime.now(UTC)
            archive_path = f"archives/events/{now.strftime('%Y%m%dT%H%M%S')}_{uuid4()}.jsonl"
            archive_text = "".join(
                json.dumps(event, ensure_ascii=True, sort_keys=True) + "\n"
                for event in events
            )
            snapshot = {
                "schema_version": HF_BUCKET_VALIDATION_SCHEMA_VERSION,
                "project_slug": project,
                "updated_at": now.isoformat(),
                "compacted_event_count": len(events),
                "items": reconciled_items,
            }
            self._api.write_files(
                bucket_id=self._bucket_id,
                token=self._token,
                files={
                    archive_path: archive_text.encode("utf-8"),
                    "snapshots/current.json": _json_bytes(snapshot),
                },
                delete_paths=active_paths,
            )
            return HfBucketCompactionResult(
                archive_path=archive_path,
                compacted_event_count=len(events),
            )

    def save_validation(self, project_slug: str, item: Validation, expected_version: int | None = None) -> int:
        project = (project_slug or "").strip()
        if not project:
            raise HfBucketValidationError("A project slug is required to save validation state.")

        with self._lock:
            self.compact_events(project)
            items = self.load_current_snapshot(project)
            current_version = int(items.get(item.detection_key, {}).get("version", 0))
            expected = expected_version if expected_version is not None else current_version
            if expected != current_version:
                raise OptimisticLockError(item.detection_key, expected, current_version)

            now = datetime.now(UTC)
            timestamp = now.isoformat()
            new_version = current_version + 1
            event_id = str(uuid4())
            event = {
                "schema_version": HF_BUCKET_VALIDATION_SCHEMA_VERSION,
                "event_id": event_id,
                "timestamp": timestamp,
                "project_slug": project,
                "expected_version": expected,
                "previous_version": current_version,
                "new_version": new_version,
                **item.model_dump(),
            }
            items[item.detection_key] = {
                "status": item.status,
                "corrected_species": item.corrected_species,
                "notes": item.notes,
                "validator": item.validator,
                "updated_at": timestamp,
                "version": new_version,
            }
            snapshot = {
                "schema_version": HF_BUCKET_VALIDATION_SCHEMA_VERSION,
                "project_slug": project,
                "updated_at": timestamp,
                "items": items,
            }
            self._api.write_files(
                bucket_id=self._bucket_id,
                token=self._token,
                files={
                    f"events/{now.strftime('%Y%m%d')}/{event_id}.json": _json_bytes(event),
                    "snapshots/current.json": _json_bytes(snapshot),
                },
            )
            return new_version
