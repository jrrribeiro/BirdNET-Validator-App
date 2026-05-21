from src.repositories.supabase_state import SupabaseBootstrapStore


class FakeSupabaseClient:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, object]]] = {
            "projects": [
                {
                    "project_id": "old-id",
                    "project_slug": "old-project",
                    "name": "Old Project",
                    "dataset_repo_id": "org/old-project",
                    "visibility": "collaborative",
                    "owner_username": "owner",
                    "dataset_token": None,
                    "active": True,
                }
            ],
            "user_project_access": [],
            "pending_invites": [],
        }
        self.patches: list[tuple[str, dict[str, object], dict[str, str]]] = []

    def select(self, table: str, *, query: dict[str, str] | None = None) -> list[dict[str, object]]:
        _ = query
        return [dict(row) for row in self.tables.get(table, [])]

    def upsert(self, table: str, rows: list[dict[str, object]], *, on_conflict: str) -> None:
        _ = on_conflict
        if table != "projects":
            self.tables[table] = [dict(row) for row in rows]
            return

        by_slug = {
            str(row.get("project_slug") or ""): row
            for row in self.tables.get("projects", [])
        }
        for row in rows:
            by_slug[str(row.get("project_slug") or "")] = dict(row)
        self.tables["projects"] = list(by_slug.values())

    def insert(self, table: str, row: dict[str, object]) -> None:
        self.tables.setdefault(table, []).append(dict(row))

    def patch(self, table: str, row: dict[str, object], *, query: dict[str, str]) -> None:
        self.patches.append((table, dict(row), dict(query)))
        if table != "projects":
            return
        project_slug_filter = str(query.get("project_slug") or "")
        if not project_slug_filter.startswith("eq."):
            return
        project_slug = project_slug_filter[3:]
        for existing in self.tables.get("projects", []):
            if existing.get("project_slug") == project_slug:
                existing.update(row)

    def delete(self, table: str, *, query: dict[str, str]) -> None:
        _ = query
        self.tables[table] = []


def test_supabase_persist_deactivates_projects_missing_from_current_state() -> None:
    client = FakeSupabaseClient()
    store = SupabaseBootstrapStore(client)  # type: ignore[arg-type]

    store.persist(
        projects=[
            {
                "project_id": "new-id",
                "project_slug": "new-project",
                "name": "New Project",
                "dataset_repo_id": "org/new-project",
                "visibility": "collaborative",
                "owner_username": "owner",
                "dataset_token": None,
                "active": True,
            }
        ],
        user_access={"owner": {"new-project": "admin"}},
        invites={},
    )

    old_project = next(row for row in client.tables["projects"] if row["project_slug"] == "old-project")

    assert old_project["active"] is False
    assert any(table == "projects" and row["active"] is False for table, row, _ in client.patches)


def test_supabase_load_projects_skips_inactive_rows() -> None:
    client = FakeSupabaseClient()
    client.tables["projects"].append(
        {
            "project_id": "active-id",
            "project_slug": "active-project",
            "name": "Active Project",
            "dataset_repo_id": "org/active-project",
            "visibility": "collaborative",
            "owner_username": "owner",
            "dataset_token": None,
            "active": True,
        }
    )
    client.tables["projects"][0]["active"] = False
    store = SupabaseBootstrapStore(client)  # type: ignore[arg-type]

    projects = store.load_projects()

    assert [project.project_slug for project in projects] == ["active-project"]
