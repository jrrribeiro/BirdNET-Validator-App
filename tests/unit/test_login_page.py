import gradio as gr

from src.auth.auth_service import AuthService
from src.domain.models import Role
from src.ui.login_page import create_login_page, oauth_action_button_html, perform_login, perform_oauth_login


def test_perform_login_blocks_username_when_disabled() -> None:
    auth_service = AuthService()
    auth_service.register_user_project_access("admin_user", {"project-a": Role.admin})

    session_id, message = perform_login(
        auth_service,
        "admin_user",
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
        allow_username_login=True,
    )

    assert session_id
    assert "Welcome, admin_user" in message


def test_perform_oauth_login_uses_verified_profile_and_token() -> None:
    class OAuthToken:
        token = "oauth_token"

    auth_service = AuthService()

    session_id, message = perform_oauth_login(
        auth_service,
        {"preferred_username": "oauth_user", "email": "oauth_user@example.org"},
        OAuthToken(),  # type: ignore[arg-type]
    )

    assert session_id
    assert "Welcome, oauth_user" in message
    assert auth_service.get_session(session_id).authentication_method == "oauth"
    assert auth_service.get_hf_token_for_user("oauth_user") == "oauth_token"
    assert auth_service.get_known_email_for_user("oauth_user") == "oauth_user@example.org"


def test_perform_oauth_login_requires_signed_in_profile() -> None:
    auth_service = AuthService()

    session_id, message = perform_oauth_login(auth_service, None, None)

    assert session_id == ""
    assert "Sign in with Hugging Face first" in message


def test_oauth_login_page_registers_gradio_oauth_routes_without_exposing_native_button() -> None:
    auth_service = AuthService()

    with gr.Blocks() as demo:
        create_login_page(auth_service, enable_oauth_login=True, admin_storage_mode=True)

    assert demo.expects_oauth is True


def test_oauth_action_button_html_has_fixed_width_and_intent_route() -> None:
    signed_out = oauth_action_button_html(signed_in=False)
    signed_in = oauth_action_button_html(signed_in=True)

    assert "max-width:var(--bn-login-width,680px)" in signed_out
    assert "/login/huggingface?birdnet_login_intent=1" in signed_out
    assert "birdnet_hf_login_intent" in signed_out
    assert "Sign out" in signed_in
    assert "/logout" in signed_in
