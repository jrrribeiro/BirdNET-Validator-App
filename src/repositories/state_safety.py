import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


class StateSafetyError(Exception):
    """Raised when a persistence operation looks unintentionally destructive."""


def _project_slugs(projects: list[dict[str, object]]) -> set[str]:
    return {
        str(project.get("project_slug") or "").strip()
        for project in projects
        if str(project.get("project_slug") or "").strip() and bool(project.get("active", True))
    }


def _access_pairs(user_access: dict[str, dict[str, object]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for username, roles in (user_access or {}).items():
        if not isinstance(roles, dict):
            continue
        normalized_username = str(username or "").strip()
        if not normalized_username:
            continue
        for project_slug in roles:
            normalized_slug = str(project_slug or "").strip()
            if normalized_slug:
                pairs.add((normalized_username, normalized_slug))
    return pairs


def assert_bootstrap_persist_is_safe(
    *,
    existing_projects: list[dict[str, object]],
    new_projects: list[dict[str, object]],
    existing_user_access: dict[str, dict[str, object]],
    new_user_access: dict[str, dict[str, object]],
    allowed_removed_project_slugs: set[str] | None = None,
) -> None:
    """Block broad project/ACL removals unless a caller declares the intent.

    Project creation, token updates, invite creation, and invite revocation should
    not remove existing project records or user access. If an app reload bug leaves
    the in-memory catalog incomplete, this guard prevents that incomplete state
    from becoming the durable source of truth.
    """
    allowed = {slug.strip() for slug in (allowed_removed_project_slugs or set()) if slug.strip()}
    existing_slugs = _project_slugs(existing_projects)
    new_slugs = _project_slugs(new_projects)
    removed_slugs = existing_slugs - new_slugs
    unexpected_removed_slugs = removed_slugs - allowed
    if unexpected_removed_slugs:
        raise StateSafetyError(
            "Refusing to persist bootstrap state because it would remove project(s) without explicit delete intent: "
            + ", ".join(sorted(unexpected_removed_slugs))
        )

    existing_access = _access_pairs(existing_user_access)
    new_access = _access_pairs(new_user_access)
    removed_access = existing_access - new_access
    unexpected_removed_access = {
        (username, project_slug)
        for username, project_slug in removed_access
        if project_slug not in allowed
    }
    if unexpected_removed_access:
        sample = ", ".join(
            f"{username}:{project_slug}"
            for username, project_slug in sorted(unexpected_removed_access)[:8]
        )
        suffix = "" if len(unexpected_removed_access) <= 8 else f" (+{len(unexpected_removed_access) - 8} more)"
        raise StateSafetyError(
            "Refusing to persist bootstrap state because it would remove user access without explicit delete intent: "
            + sample
            + suffix
        )


def load_json_object(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _backup_existing_file(path: Path, *, max_backups: int) -> None:
    if not path.exists() or not path.is_file():
        return

    backup_dir = path.parent / ".backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    backup_path = backup_dir / f"{path.name}.{timestamp}.{uuid4().hex[:8]}.bak"
    shutil.copy2(path, backup_path)

    backups = sorted(backup_dir.glob(f"{path.name}.*.bak"), key=lambda item: item.name)
    excess = len(backups) - max(0, int(max_backups))
    if excess <= 0:
        return
    for old_backup in backups[:excess]:
        try:
            old_backup.unlink()
        except OSError:
            pass


def atomic_write_json_with_backup(path: Path, payload: object, *, max_backups: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _backup_existing_file(path, max_backups=max_backups)
    tmp_path = path.parent / f".{path.name}.tmp.{os.getpid()}.{uuid4().hex}"
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
