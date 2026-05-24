import json

import pytest

from src.domain.models import Project
from src.services.hf_project_state_store import (
    HF_PROJECT_STATE_BACKEND,
    HfProjectStateStoreError,
    HfProjectStateStoreInitializer,
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
