import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from src.domain.models import Validation
from src.repositories.append_only_validation_repository import OptimisticLockError
from src.services.hf_project_state_store import HF_PROJECT_STATE_SCHEMA_VERSION


class HfProjectStateValidationError(Exception):
    """Raised when the Hugging Face project state validation backend cannot continue safely."""


class HfProjectStateFilesApi(Protocol):
    def list_repo_files(
        self,
        repo_id: str,
        *,
        repo_type: str | None = None,
        token: str | None = None,
    ) -> list[str]: ...

    def read_text(
        self,
        repo_id: str,
        path_in_repo: str,
        *,
        repo_type: str | None = None,
        token: str | None = None,
    ) -> str: ...

    def create_commit(
        self,
        repo_id: str,
        operations: list[CommitOperationAdd],
        *,
        commit_message: str,
        token: str | None = None,
        repo_type: str | None = None,
    ) -> object: ...


class HuggingFaceProjectStateFilesApi:
    def __init__(self, api: HfApi | None = None) -> None:
        self._api = api or HfApi()

    def list_repo_files(
        self,
        repo_id: str,
        *,
        repo_type: str | None = None,
        token: str | None = None,
    ) -> list[str]:
        return self._api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token)

    def read_text(
        self,
        repo_id: str,
        path_in_repo: str,
        *,
        repo_type: str | None = None,
        token: str | None = None,
    ) -> str:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            repo_type=repo_type,
            token=token,
        )
        return Path(local_path).read_text(encoding="utf-8")

    def create_commit(
        self,
        repo_id: str,
        operations: list[CommitOperationAdd],
        *,
        commit_message: str,
        token: str | None = None,
        repo_type: str | None = None,
    ) -> object:
        return self._api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=commit_message,
            token=token,
            repo_type=repo_type,
        )


@dataclass(frozen=True)
class HfProjectStateValidationRef:
    state_repo_id: str
    token: str


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True).encode("utf-8")


def _normalize_snapshot_item(item: object) -> dict[str, object]:
    payload = dict(item) if isinstance(item, dict) else {}
    return {
        "status": payload.get("status"),
        "corrected_species": payload.get("corrected_species"),
        "notes": payload.get("notes") or "",
        "validator": payload.get("validator"),
        "updated_at": payload.get("updated_at") or payload.get("timestamp") or payload.get("created_at"),
        "version": int(payload.get("version") or payload.get("new_version") or 0),
    }


def _extract_snapshot_items(payload: object, project_slug: str) -> dict[str, dict[str, object]] | None:
    if not isinstance(payload, dict):
        return None

    if "items" in payload:
        if str(payload.get("project_slug") or project_slug) != project_slug:
            return None
        raw_items = payload.get("items")
    else:
        raw_items = payload

    if not isinstance(raw_items, dict):
        return None

    return {
        str(detection_key): _normalize_snapshot_item(item)
        for detection_key, item in raw_items.items()
        if str(detection_key).strip()
    }


class HfProjectStateValidationRepository:
    def __init__(
        self,
        *,
        state_repo_id: str,
        token: str,
        api: HfProjectStateFilesApi | None = None,
    ) -> None:
        self._state_repo_id = (state_repo_id or "").strip()
        self._token = (token or "").strip()
        self._api = api or HuggingFaceProjectStateFilesApi()
        self._lock = threading.RLock()

        if not self._state_repo_id:
            raise HfProjectStateValidationError("A project state repo id is required.")
        if not self._token:
            raise HfProjectStateValidationError("A Hugging Face token is required to read and write project state.")

    def _read_text_or_none(self, path_in_repo: str) -> str | None:
        try:
            return self._api.read_text(
                repo_id=self._state_repo_id,
                path_in_repo=path_in_repo,
                repo_type="dataset",
                token=self._token,
            )
        except (EntryNotFoundError, FileNotFoundError):
            return None

    def _read_json_or_none(self, path_in_repo: str) -> object | None:
        text = self._read_text_or_none(path_in_repo)
        if text is None or not text.strip():
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _event_payloads_from_text(self, path_in_repo: str, text: str) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        if path_in_repo.endswith(".jsonl"):
            lines = text.splitlines()
        else:
            lines = [text]

        for line in lines:
            payload = line.strip()
            if not payload:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def list_events(self, project_slug: str) -> list[dict[str, object]]:
        project = (project_slug or "").strip()
        files = self._api.list_repo_files(
            repo_id=self._state_repo_id,
            repo_type="dataset",
            token=self._token,
        )
        events: list[dict[str, object]] = []
        for path_in_repo in sorted(files):
            if not path_in_repo.startswith("events/") or not path_in_repo.endswith((".json", ".jsonl")):
                continue
            text = self._read_text_or_none(path_in_repo)
            if text is None:
                continue
            for event in self._event_payloads_from_text(path_in_repo, text):
                event_project = str(event.get("project_slug") or project).strip()
                if project and event_project != project:
                    continue
                events.append(event)

        return sorted(events, key=lambda event: str(event.get("timestamp") or event.get("created_at") or ""))

    def _snapshot_from_events(self, project_slug: str) -> dict[str, dict[str, object]]:
        snapshot: dict[str, dict[str, object]] = {}
        for event in self.list_events(project_slug):
            detection_key = str(event.get("detection_key") or "").strip()
            if not detection_key:
                continue
            snapshot[detection_key] = {
                "status": event.get("status"),
                "corrected_species": event.get("corrected_species"),
                "notes": event.get("notes") or "",
                "validator": event.get("validator"),
                "updated_at": event.get("timestamp") or event.get("created_at"),
                "version": int(event.get("new_version") or event.get("version") or 0),
            }
        return snapshot

    def load_current_snapshot(self, project_slug: str) -> dict[str, dict[str, object]]:
        with self._lock:
            payload = self._read_json_or_none("snapshots/current.json")
            items = _extract_snapshot_items(payload, project_slug)
            if items is not None:
                return items
            return self._snapshot_from_events(project_slug)

    def save_validation(self, project_slug: str, item: Validation, expected_version: int | None = None) -> int:
        project = (project_slug or "").strip()
        if not project:
            raise HfProjectStateValidationError("A project slug is required to save validation state.")

        with self._lock:
            current_payload = self.load_current_snapshot(project)
            current_item = current_payload.get(item.detection_key, {})
            current_version = int(current_item.get("version", 0))
            expected = expected_version if expected_version is not None else current_version
            if expected != current_version:
                raise OptimisticLockError(
                    detection_key=item.detection_key,
                    expected_version=expected,
                    current_version=current_version,
                )

            new_version = current_version + 1
            now = datetime.now(UTC)
            timestamp = now.isoformat()
            event_id = str(uuid4())
            event = {
                "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
                "event_id": event_id,
                "timestamp": timestamp,
                "project_slug": project,
                "expected_version": expected,
                "previous_version": current_version,
                "new_version": new_version,
                **item.model_dump(),
            }
            current_payload[item.detection_key] = {
                "status": item.status,
                "corrected_species": item.corrected_species,
                "notes": item.notes,
                "validator": item.validator,
                "updated_at": timestamp,
                "version": new_version,
            }
            snapshot = {
                "schema_version": HF_PROJECT_STATE_SCHEMA_VERSION,
                "project_slug": project,
                "updated_at": timestamp,
                "items": current_payload,
            }
            event_path = f"events/{now.strftime('%Y%m%d')}/{event_id}.json"

            self._api.create_commit(
                repo_id=self._state_repo_id,
                repo_type="dataset",
                token=self._token,
                commit_message=f"Validate detection {item.detection_key[:32]}",
                operations=[
                    CommitOperationAdd(
                        path_in_repo=event_path,
                        path_or_fileobj=_json_bytes(event),
                    ),
                    CommitOperationAdd(
                        path_in_repo="snapshots/current.json",
                        path_or_fileobj=_json_bytes(snapshot),
                    ),
                ],
            )
            return new_version
