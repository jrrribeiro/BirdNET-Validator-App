import json

import pytest

from src.domain.models import Project
from src.services.hf_project_state_store import (
    HF_PROJECT_STATE_BACKEND,
    HfProjectStateStoreError,
    HfProjectStateStoreConnector,
    HfProjectStateStoreInitializer,
    HfProjectStatePermissionProbe,
    HfProjectStateStoreLoadedProject,
    HfProjectStateStoreLoader,
    HfProjectStateStoreSync,
    default_state_repo_id,
)


class FakeHfProjectStateApi:
    def __init__(self, existing_files: list[str] | None = None) -> None:
        self.created: list[dict[str, object]] = []
        self.commits: list[dict[str, object]] = []
        self.visibility_updates: list[dict[str, object]] = []
        self.existing_files = existing_files or []

    def create_repo(self, **kwargs):  # noqa: ANN001
        self.created.append(dict(kwargs))
        return object()

    def create_commit(self, **kwargs):  # noqa: ANN001
        self.commits.append(dict(kwargs))
        return object()

    def list_repo_files(self, **kwargs):  # noqa: ANN001
        return list(self.existing_files)

    def update_repo_visibility(self, **kwargs):  # noqa: ANN001
        self.visibility_updates.append(dict(kwargs))
        return {"private": True}


class FakeHfProjectStateReadApi:
    def __init__(self, files: dict[str, str]) -> None:
        self.files = dict(files)

    def read_text(self, **kwargs):  # noqa: ANN001
        path = str(kwargs["path_in_repo"])
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


def _project() -> Project:
    return Project(
        project_slug="upload-test2",
        name="Upload Test 2",
        dataset_repo_id="jrrribeiro/upload_test2",
        owner_username="jrrribeiro",
    )


def test_default_state_repo_id_uses_dataset_namespace_and_state_suffix() -> None:
    assert default_state_repo_id("jrrribeiro/upload_test2") == "jrrribeiro/upload_test2_state"


def test_default_state_repo_id_rejects_invalid_dataset_repo() -> None:
    with pytest.raises(HfProjectStateStoreError):
        default_state_repo_id("upload_test2")


def test_initialize_creates_private_dataset_and_manifest_commit() -> None:
    fake_api = FakeHfProjectStateApi()
    initializer = HfProjectStateStoreInitializer(api=fake_api)

    result = initializer.initialize(
        project=_project(),
        creator_username="jrrribeiro",
        token="hf_test",
    )

    assert result.state_repo_id == "jrrribeiro/upload_test2_state"
    assert result.manifest["state_backend"] == HF_PROJECT_STATE_BACKEND
    assert result.initialized is True
    assert result.reused_existing is False
    assert fake_api.created == [
        {
            "repo_id": "jrrribeiro/upload_test2_state",
            "token": "hf_test",
            "private": True,
            "repo_type": "dataset",
            "exist_ok": True,
        }
    ]
    assert fake_api.visibility_updates == [
        {
            "repo_id": "jrrribeiro/upload_test2_state",
            "token": "hf_test",
            "private": True,
            "repo_type": "dataset",
        }
    ]
    commit = fake_api.commits[0]
    assert commit["repo_id"] == "jrrribeiro/upload_test2_state"
    assert commit["repo_type"] == "dataset"
    paths = [operation.path_in_repo for operation in commit["operations"]]
    assert paths == [
        "README.md",
        "project.json",
        "acl.json",
        "invites.json",
        "snapshots/current.json",
        "events/.gitkeep",
    ]
    project_payload = json.loads(commit["operations"][1].path_or_fileobj.decode("utf-8"))
    acl_payload = json.loads(commit["operations"][2].path_or_fileobj.decode("utf-8"))
    assert project_payload["state_ref"] == "jrrribeiro/upload_test2_state"
    assert acl_payload["users"]["jrrribeiro"]["role"] == "admin"


def test_initialize_reuses_existing_manifest_without_overwriting_state() -> None:
    fake_api = FakeHfProjectStateApi(existing_files=["README.md", "project.json", "snapshots/current.json"])
    initializer = HfProjectStateStoreInitializer(api=fake_api)

    result = initializer.initialize(
        project=_project(),
        creator_username="jrrribeiro",
        token="hf_test",
    )

    assert result.state_repo_id == "jrrribeiro/upload_test2_state"
    assert result.initialized is False
    assert result.reused_existing is True
    assert fake_api.commits == []


def test_initialize_completes_repo_created_with_hub_scaffolding_only() -> None:
    fake_api = FakeHfProjectStateApi(existing_files=[".gitattributes"])
    initializer = HfProjectStateStoreInitializer(api=fake_api)

    result = initializer.initialize(
        project=_project(),
        creator_username="jrrribeiro",
        token="hf_test",
    )

    assert result.initialized is True
    assert result.reused_existing is False
    assert len(fake_api.commits) == 1
    assert "project.json" in [operation.path_in_repo for operation in fake_api.commits[0]["operations"]]


def test_initialize_refuses_nonempty_repo_without_manifest() -> None:
    fake_api = FakeHfProjectStateApi(existing_files=["README.md", "events/event.jsonl"])
    initializer = HfProjectStateStoreInitializer(api=fake_api)

    with pytest.raises(HfProjectStateStoreError, match="Refusing to initialize automatically"):
        initializer.initialize(
            project=_project(),
            creator_username="jrrribeiro",
            token="hf_test",
        )

    assert fake_api.commits == []


def test_initialize_requires_token() -> None:
    initializer = HfProjectStateStoreInitializer(api=FakeHfProjectStateApi())

    with pytest.raises(HfProjectStateStoreError):
        initializer.initialize(project=_project(), creator_username="jrrribeiro", token="")


def test_sync_project_state_writes_project_acl_and_invites_without_token_values() -> None:
    fake_api = FakeHfProjectStateApi()
    sync = HfProjectStateStoreSync(api=fake_api)
    project = _project()
    project.state_repo_id = "jrrribeiro/upload_test2_state"
    project.state_backend = HF_PROJECT_STATE_BACKEND
    project.dataset_token = "hf_should_not_be_serialized"

    result = sync.sync_project_state(
        project=project,
        user_access={
            "jrrribeiro": {"upload-test2": "admin"},
            "validator-a": {"upload-test2": "validator"},
            "other": {"another-project": "admin"},
        },
        pending_invites={
            "validator-b": {
                "upload-test2": {
                    "role": "validator",
                    "invited_by": "jrrribeiro",
                    "created_at": "2026-05-24T12:00:00+00:00",
                    "expires_at": "2026-05-25T12:00:00+00:00",
                    "username": "validator-b",
                    "invitee_email": "",
                }
            },
            "other": {"another-project": {"role": "validator"}},
        },
        token="hf_write",
        actor_username="jrrribeiro",
    )

    assert result.state_repo_id == "jrrribeiro/upload_test2_state"
    commit = fake_api.commits[0]
    assert commit["repo_id"] == "jrrribeiro/upload_test2_state"
    paths = [operation.path_in_repo for operation in commit["operations"]]
    assert paths == ["project.json", "acl.json", "invites.json"]
    project_payload = json.loads(commit["operations"][0].path_or_fileobj.decode("utf-8"))
    acl_payload = json.loads(commit["operations"][1].path_or_fileobj.decode("utf-8"))
    invites_payload = json.loads(commit["operations"][2].path_or_fileobj.decode("utf-8"))
    serialized = json.dumps([project_payload, acl_payload, invites_payload])
    assert "hf_should_not_be_serialized" not in serialized
    assert sorted(acl_payload["users"]) == ["jrrribeiro", "validator-a"]
    assert acl_payload["users"]["validator-a"]["role"] == "validator"
    assert sorted(invites_payload["pending"]) == ["validator-b"]


def test_sync_project_state_archives_without_deleting_validation_files() -> None:
    fake_api = FakeHfProjectStateApi()
    sync = HfProjectStateStoreSync(api=fake_api)
    project = _project()
    project.state_repo_id = "jrrribeiro/upload_test2_state"
    project.active = False

    sync.sync_project_state(
        project=project,
        user_access={"jrrribeiro": {"upload-test2": "admin"}},
        pending_invites={"validator-b": {"upload-test2": {"role": "validator"}}},
        token="hf_write",
        actor_username="jrrribeiro",
        archived=True,
    )

    commit = fake_api.commits[0]
    paths = [operation.path_in_repo for operation in commit["operations"]]
    assert paths == ["project.json", "acl.json", "invites.json"]
    project_payload = json.loads(commit["operations"][0].path_or_fileobj.decode("utf-8"))
    acl_payload = json.loads(commit["operations"][1].path_or_fileobj.decode("utf-8"))
    invites_payload = json.loads(commit["operations"][2].path_or_fileobj.decode("utf-8"))
    assert project_payload["active"] is False
    assert project_payload["state_status"] == "archived"
    assert "archived_at" in project_payload
    assert acl_payload["users"] == {}
    assert invites_payload["pending"] == {}


def test_sync_project_state_requires_token() -> None:
    sync = HfProjectStateStoreSync(api=FakeHfProjectStateApi())

    with pytest.raises(HfProjectStateStoreError):
        sync.sync_project_state(
            project=_project(),
            user_access={},
            pending_invites={},
            token="",
        )


def test_permission_probe_writes_isolated_diagnostic_with_acting_token_only() -> None:
    fake_api = FakeHfProjectStateApi()
    project = _project()
    project.state_repo_id = "jrrribeiro/upload_test2_state"

    class FakeLoader:
        def load_project_state(self, *, state_repo_id: str, token: str):  # noqa: ANN001
            assert state_repo_id == project.state_repo_id
            assert token == "hf_validator"
            return HfProjectStateStoreLoadedProject(
                state_repo_id=state_repo_id,
                project=project,
                user_access={},
                pending_invites={},
            )

    probe = HfProjectStatePermissionProbe(loader=FakeLoader(), api=fake_api)  # type: ignore[arg-type]
    result = probe.probe(project=project, actor_username="validator-a", token="hf_validator")

    assert result.actor_username == "validator-a"
    assert result.diagnostic_path.startswith("diagnostics/oauth-permission-proof/")
    commit = fake_api.commits[0]
    assert commit["token"] == "hf_validator"
    assert commit["operations"][0].path_in_repo == result.diagnostic_path
    serialized = commit["operations"][0].path_or_fileobj.decode("utf-8")
    assert "validator-a" in serialized
    assert "hf_validator" not in serialized


def test_permission_probe_requires_personal_token() -> None:
    project = _project()
    project.state_repo_id = "jrrribeiro/upload_test2_state"
    probe = HfProjectStatePermissionProbe(api=FakeHfProjectStateApi())

    with pytest.raises(HfProjectStateStoreError, match="OAuth"):
        probe.probe(project=project, actor_username="validator-a", token="")


def test_load_project_state_recovers_project_acl_and_invites() -> None:
    loader = HfProjectStateStoreLoader(
        api=FakeHfProjectStateReadApi(
            {
                "project.json": json.dumps(
                    {
                        "schema_version": 1,
                        "project_id": "project-state-id",
                        "project_slug": "upload-test2",
                        "project_name": "Upload Test 2",
                        "dataset_repo_id": "jrrribeiro/upload_test2",
                        "state_status": "ready",
                        "owner_username": "jrrribeiro",
                        "visibility": "collaborative",
                        "active": True,
                    }
                ),
                "acl.json": json.dumps(
                    {
                        "schema_version": 1,
                        "project_slug": "upload-test2",
                        "users": {
                            "jrrribeiro": {"role": "admin", "active": True},
                            "validator-a": {"role": "validator", "active": True},
                            "inactive": {"role": "validator", "active": False},
                        },
                    }
                ),
                "invites.json": json.dumps(
                    {
                        "schema_version": 1,
                        "project_slug": "upload-test2",
                        "pending": {
                            "validator-b": {
                                "role": "validator",
                                "invited_by": "jrrribeiro",
                                "created_at": "2026-05-24T12:00:00+00:00",
                                "expires_at": "2026-05-25T12:00:00+00:00",
                                "username": "validator-b",
                                "invitee_email": "",
                            }
                        },
                    }
                ),
            }
        )
    )

    loaded = loader.load_project_state(state_repo_id="jrrribeiro/upload_test2_state", token="hf_read")

    assert loaded is not None
    assert loaded.project.project_slug == "upload-test2"
    assert loaded.project.state_backend == HF_PROJECT_STATE_BACKEND
    assert loaded.project.state_repo_id == "jrrribeiro/upload_test2_state"
    assert loaded.user_access["jrrribeiro"]["upload-test2"].value == "admin"
    assert loaded.user_access["validator-a"]["upload-test2"].value == "validator"
    assert "inactive" not in loaded.user_access
    assert loaded.pending_invites["validator-b"]["upload-test2"]["role"] == "validator"


def test_load_project_state_skips_archived_project() -> None:
    loader = HfProjectStateStoreLoader(
        api=FakeHfProjectStateReadApi(
            {
                "project.json": json.dumps(
                    {
                        "schema_version": 1,
                        "project_slug": "upload-test2",
                        "project_name": "Upload Test 2",
                        "dataset_repo_id": "jrrribeiro/upload_test2",
                        "state_status": "archived",
                        "active": False,
                    }
                )
            }
        )
    )

    assert loader.load_project_state(state_repo_id="jrrribeiro/upload_test2_state", token="hf_read") is None


def _connector_loader(*, visibility: str = "collaborative", owner: str = "jrrribeiro") -> HfProjectStateStoreLoader:
    return HfProjectStateStoreLoader(
        api=FakeHfProjectStateReadApi(
            {
                "project.json": json.dumps(
                    {
                        "schema_version": 1,
                        "project_slug": "upload-test2",
                        "project_name": "Upload Test 2",
                        "dataset_repo_id": "jrrribeiro/upload_test2",
                        "state_status": "ready",
                        "owner_username": owner,
                        "visibility": visibility,
                        "active": True,
                    }
                ),
                "acl.json": json.dumps(
                    {
                        "schema_version": 1,
                        "project_slug": "upload-test2",
                        "users": {
                            "jrrribeiro": {"role": "admin", "active": True},
                            "validator-a": {"role": "validator", "active": True},
                        },
                    }
                ),
                "invites.json": json.dumps({"schema_version": 1, "project_slug": "upload-test2", "pending": {}}),
            }
        )
    )


def test_connector_allows_acl_admin_with_authenticated_token() -> None:
    connector = HfProjectStateStoreConnector(loader=_connector_loader())

    loaded = connector.connect_admin_project(
        state_repo_id="jrrribeiro/upload_test2_state",
        token="hf_admin",
        actor_username="jrrribeiro",
    )

    assert loaded.project.project_slug == "upload-test2"


def test_connector_rejects_validator_or_missing_authenticated_token() -> None:
    connector = HfProjectStateStoreConnector(loader=_connector_loader())

    with pytest.raises(HfProjectStateStoreError, match="ADMIN"):
        connector.connect_admin_project(
            state_repo_id="jrrribeiro/upload_test2_state",
            token="hf_validator",
            actor_username="validator-a",
        )

    with pytest.raises(HfProjectStateStoreError, match="Sign in"):
        connector.connect_admin_project(
            state_repo_id="jrrribeiro/upload_test2_state",
            token="",
            actor_username="jrrribeiro",
        )


def test_connector_restricts_private_project_connection_to_owner() -> None:
    connector = HfProjectStateStoreConnector(loader=_connector_loader(visibility="private", owner="owner-account"))

    with pytest.raises(HfProjectStateStoreError, match="Only the owner"):
        connector.connect_admin_project(
            state_repo_id="jrrribeiro/upload_test2_state",
            token="hf_admin",
            actor_username="jrrribeiro",
        )
