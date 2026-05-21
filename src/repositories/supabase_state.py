import json
import threading
from datetime import UTC, datetime
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

from src.domain.models import Project, Role, Validation
from src.repositories.append_only_validation_repository import OptimisticLockError


class SupabaseStateError(Exception):
    pass


class SupabaseRestClient:
    def __init__(self, url: str, service_role_key: str, timeout_seconds: int = 20) -> None:
        self._url = url.rstrip("/")
        self._key = service_role_key
        self._timeout_seconds = timeout_seconds

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        payload: object | None = None,
        prefer: str | None = None,
    ) -> object:
        query_string = f"?{urlencode(query)}" if query else ""
        request = Request(
            f"{self._url}/rest/v1/{path}{query_string}",
            method=method,
            headers={
                "apikey": self._key,
                "Authorization": f"Bearer {self._key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                **({"Prefer": prefer} if prefer else {}),
            },
            data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise SupabaseStateError(f"Supabase HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise SupabaseStateError(f"Supabase connection error: {exc}") from exc

        if not body.strip():
            return None
        return json.loads(body)

    @staticmethod
    def eq(value: str) -> str:
        return f"eq.{quote(value, safe='')}"

    def select(self, table: str, *, query: dict[str, str] | None = None) -> list[dict[str, object]]:
        payload = self._request("GET", table, query={"select": "*", **(query or {})})
        return payload if isinstance(payload, list) else []

    def upsert(self, table: str, rows: list[dict[str, object]], *, on_conflict: str) -> None:
        if not rows:
            return
        self._request(
            "POST",
            table,
            query={"on_conflict": on_conflict},
            payload=rows,
            prefer="resolution=merge-duplicates",
        )

    def insert(self, table: str, row: dict[str, object]) -> None:
        self._request("POST", table, payload=row)

    def patch(self, table: str, row: dict[str, object], *, query: dict[str, str]) -> None:
        self._request("PATCH", table, query=query, payload=row)

    def delete(self, table: str, *, query: dict[str, str]) -> None:
        self._request("DELETE", table, query=query)


class SupabaseBootstrapStore:
    def __init__(self, client: SupabaseRestClient) -> None:
        self._client = client

    def load_projects(self) -> list[Project]:
        projects: list[Project] = []
        for row in self._client.select("projects", query={"order": "project_slug.asc"}):
            if not bool(row.get("active", True)):
                continue
            try:
                projects.append(
                    Project(
                        project_id=str(row.get("project_id") or uuid4()),
                        project_slug=str(row.get("project_slug") or "").strip(),
                        name=str(row.get("name") or "").strip(),
                        dataset_repo_id=str(row.get("dataset_repo_id") or "").strip(),
                        visibility=str(row.get("visibility") or "collaborative").strip(),
                        owner_username=(str(row.get("owner_username") or "").strip() or None),
                        dataset_token=(str(row.get("dataset_token") or "").strip() or None),
                        active=bool(row.get("active", True)),
                    )
                )
            except Exception:
                continue
        return projects

    def load_user_access(self) -> dict[str, dict[str, Role]]:
        access: dict[str, dict[str, Role]] = {}
        rows = self._client.select("user_project_access", query={"active": "eq.true"})
        for row in rows:
            username = str(row.get("username") or "").strip()
            project_slug = str(row.get("project_slug") or "").strip()
            role_text = str(row.get("role") or "").strip().lower()
            if not username or not project_slug or role_text not in {"admin", "validator"}:
                continue
            access.setdefault(username, {})[project_slug] = Role(role_text)
        return access

    def load_pending_invites(self) -> dict[str, dict[str, dict[str, str]]]:
        payload: dict[str, dict[str, dict[str, str]]] = {}
        rows = self._client.select("pending_invites", query={"status": "eq.pending"})
        for row in rows:
            project_slug = str(row.get("project_slug") or "").strip()
            role = str(row.get("role") or "").strip().lower()
            username = str(row.get("username") or "").strip()
            email = str(row.get("email") or "").strip()
            if not project_slug or role not in {"admin", "validator"} or (not username and not email):
                continue
            key = username or f"email:{email}"
            payload.setdefault(key, {})[project_slug] = {
                "role": role,
                "invited_by": str(row.get("invited_by") or "admin").strip() or "admin",
                "created_at": str(row.get("created_at") or ""),
                "expires_at": str(row.get("expires_at") or ""),
                "username": username,
                "invitee_email": email,
            }
        return payload

    def persist(self, projects: list[dict[str, object]], user_access: dict[str, dict[str, str]], invites: dict[str, dict[str, dict[str, str]]]) -> None:
        project_rows = [
            {
                "project_id": str(project.get("project_id") or uuid4()),
                "project_slug": str(project.get("project_slug") or "").strip(),
                "name": str(project.get("name") or "").strip(),
                "dataset_repo_id": str(project.get("dataset_repo_id") or "").strip(),
                "visibility": str(project.get("visibility") or "collaborative").strip(),
                "owner_username": project.get("owner_username") or None,
                "dataset_token": project.get("dataset_token") or None,
                "active": bool(project.get("active", True)),
                "updated_at": datetime.now(UTC).isoformat(),
            }
            for project in projects
            if str(project.get("project_slug") or "").strip()
        ]
        self._client.upsert("projects", project_rows, on_conflict="project_slug")

        active_project_slugs = {
            str(project.get("project_slug") or "").strip()
            for project in projects
            if str(project.get("project_slug") or "").strip()
        }
        for row in self._client.select("projects"):
            project_slug = str(row.get("project_slug") or "").strip()
            if project_slug and project_slug not in active_project_slugs:
                self._client.patch(
                    "projects",
                    {"active": False, "updated_at": datetime.now(UTC).isoformat()},
                    query={"project_slug": SupabaseRestClient.eq(project_slug)},
                )

        access_rows = []
        seen_access: set[tuple[str, str]] = set()
        for username, roles in user_access.items():
            for project_slug, role in roles.items():
                seen_access.add((username, project_slug))
                access_rows.append(
                    {
                        "username": username,
                        "project_slug": project_slug,
                        "role": role,
                        "active": True,
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )
        self._client.upsert("user_project_access", access_rows, on_conflict="username,project_slug")

        for row in self._client.select("user_project_access"):
            key = (str(row.get("username") or ""), str(row.get("project_slug") or ""))
            if key not in seen_access and key != ("", ""):
                self._client.patch(
                    "user_project_access",
                    {"active": False, "updated_at": datetime.now(UTC).isoformat()},
                    query={"username": SupabaseRestClient.eq(key[0]), "project_slug": SupabaseRestClient.eq(key[1])},
                )

        self._client.delete("pending_invites", query={"status": "eq.pending"})
        invite_rows = []
        for invite_key, invites_by_project in invites.items():
            for project_slug, invite in invites_by_project.items():
                username = str(invite.get("username") or "").strip()
                email = str(invite.get("invitee_email") or "").strip()
                if not username and not email:
                    if invite_key.startswith("email:"):
                        email = invite_key[len("email:") :]
                    else:
                        username = invite_key
                invite_rows.append(
                    {
                        "username": username or None,
                        "email": email or None,
                        "project_slug": project_slug,
                        "role": str(invite.get("role") or "validator"),
                        "invited_by": str(invite.get("invited_by") or "admin"),
                        "status": "pending",
                        "created_at": str(invite.get("created_at") or datetime.now(UTC).isoformat()),
                        "expires_at": str(invite.get("expires_at") or ""),
                    }
                )
        if invite_rows:
            self._client.insert("pending_invites", invite_rows[0])
            if len(invite_rows) > 1:
                self._client._request("POST", "pending_invites", payload=invite_rows[1:])


class SupabaseValidationRepository:
    def __init__(self, client: SupabaseRestClient) -> None:
        self._client = client
        self._lock = threading.Lock()

    def save_validation(self, project_slug: str, item: Validation, expected_version: int | None = None) -> int:
        with self._lock:
            rows = self._client.select(
                "current_validations",
                query={
                    "project_slug": SupabaseRestClient.eq(project_slug),
                    "detection_key": SupabaseRestClient.eq(item.detection_key),
                    "limit": "1",
                },
            )
            current = rows[0] if rows else {}
            current_version = int(current.get("version") or 0)
            expected = expected_version if expected_version is not None else current_version
            if expected != current_version:
                raise OptimisticLockError(item.detection_key, expected, current_version)

            new_version = current_version + 1
            now = datetime.now(UTC).isoformat()
            event = {
                "project_slug": project_slug,
                "detection_key": item.detection_key,
                "status": item.status,
                "corrected_species": item.corrected_species,
                "notes": item.notes,
                "validator": item.validator,
                "expected_version": expected,
                "previous_version": current_version,
                "new_version": new_version,
                "created_at": now,
            }
            self._client.insert("validation_events", event)
            self._client.upsert(
                "current_validations",
                [
                    {
                        "project_slug": project_slug,
                        "detection_key": item.detection_key,
                        "status": item.status,
                        "corrected_species": item.corrected_species,
                        "notes": item.notes,
                        "validator": item.validator,
                        "version": new_version,
                        "updated_at": now,
                    }
                ],
                on_conflict="project_slug,detection_key",
            )
            return new_version

    def load_current_snapshot(self, project_slug: str) -> dict[str, dict[str, object]]:
        rows = self._client.select("current_validations", query={"project_slug": SupabaseRestClient.eq(project_slug)})
        snapshot: dict[str, dict[str, object]] = {}
        for row in rows:
            detection_key = str(row.get("detection_key") or "").strip()
            if not detection_key:
                continue
            snapshot[detection_key] = {
                "status": row.get("status"),
                "corrected_species": row.get("corrected_species"),
                "notes": row.get("notes") or "",
                "validator": row.get("validator"),
                "updated_at": row.get("updated_at"),
                "version": int(row.get("version") or 0),
            }
        return snapshot

    def list_events(self, project_slug: str) -> list[dict[str, object]]:
        rows = self._client.select(
            "validation_events",
            query={"project_slug": SupabaseRestClient.eq(project_slug), "order": "created_at.asc"},
        )
        events: list[dict[str, object]] = []
        for row in rows:
            event = dict(row)
            event["timestamp"] = event.get("created_at")
            events.append(event)
        return events
