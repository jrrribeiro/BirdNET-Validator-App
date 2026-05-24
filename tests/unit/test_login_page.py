from src.auth.auth_service import AuthService
from src.domain.models import Role
from src.ui.login_page import perform_login


def test_perform_login_blocks_username_when_disabled() -> None:
    auth_service = AuthService()
    auth_service.register_user_project_access("admin_user", {"project-a": Role.admin})

    session_id, message = perform_login(
        auth_service,
        "admin_user",
        "",
        allow_username_login=False,
    )

    assert session_id == ""
    assert "Username-only login is disabled" in message


def test_perform_login_allows_username_when_enabled() -> None:
    auth_service = AuthService()
    auth_service.register_user_project_access("admin_user", {"project-a": Role.admin})

    session_id, message = perform_login(
        auth_service,
        "admin_user",
        "",
        allow_username_login=True,
    )

    assert session_id
    assert "Welcome, admin_user" in message


def test_perform_login_allows_hf_token_when_username_disabled(monkeypatch) -> None:  # noqa: ANN001
    class FakeApi:
        def whoami(self, token: str):
            assert token == "hf_valid"
            return {"name": "hf_user", "email": "hf_user@example.org"}

    monkeypatch.setattr("src.auth.auth_service.HfApi", lambda: FakeApi())
    auth_service = AuthService()

    session_id, message = perform_login(
        auth_service,
        "",
        "hf_valid",
        allow_username_login=False,
    )

    assert session_id
    assert "Welcome, hf_user" in message
