"""Gradio login page component for multi-project authorization."""

from typing import Tuple

import gradio as gr

from src.auth.auth_service import AuthService


def perform_login(
    auth_service: AuthService,
    username: str,
    *,
    allow_username_login: bool = True,
) -> Tuple[str, str]:
    """Attempt a development-only username login and return its session status."""
    if not allow_username_login:
        return (
            "",
            "Username-only login is disabled for this deployment. Sign in with Hugging Face OAuth.",
        )

    if not username or not username.strip():
        return "", "Please enter a username"

    username = username.strip()
    session = auth_service.login(username)

    if session is None:
        return "", f"User '{username}' not found or inactive"

    admin_projects = 0
    validator_projects = 0
    for project_slug in session.authorized_projects:
        project_role = auth_service.get_user_role_for_project(username, project_slug)
        if project_role is None:
            continue
        if project_role.value == "admin":
            admin_projects += 1
        else:
            validator_projects += 1

    return (
        session.session_id,
        (
            f"Welcome, {username}. "
            f"Admin in {admin_projects} project(s), validator in {validator_projects} project(s)."
        ),
    )


def perform_oauth_login(
    auth_service: AuthService,
    profile: gr.OAuthProfile | None,
    oauth_token: gr.OAuthToken | None,
) -> Tuple[str, str]:
    """Create an app session using the verified identity supplied by HF Spaces OAuth."""
    if profile is None or oauth_token is None:
        return "", "Sign in with Hugging Face first, then continue to the workspace."

    username = str(profile.get("preferred_username") or profile.get("name") or "").strip()
    email = str(profile.get("email") or "").strip() or None
    session, message = auth_service.login_with_verified_hf_identity(
        username=username,
        token=oauth_token.token,
        email=email,
        authentication_method="oauth",
    )
    if session is None:
        return "", message
    return session.session_id, message


def create_login_page(
    auth_service: AuthService,
    *,
    allow_username_login: bool = True,
    enable_oauth_login: bool = False,
    auth_mode_label: str = "",
    admin_storage_mode: bool = False,
) -> Tuple[gr.Textbox, gr.Markdown]:
    """Create an OAuth production login with an optional local username fallback.

    Args:
        auth_service: AuthService instance for login validation

    Returns:
        Tuple of (session_output, error_message)
    """
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("")
        with gr.Column(scale=6):
            gr.Markdown("# BirdNET Validation Platform")

            session_output = gr.Textbox(
                label="Session ID",
                interactive=False,
                visible=False,
            )

            username_input = None
            login_button = None
            if enable_oauth_login:
                gr.Markdown(
                    (
                        "Secure login with your Hugging Face account. The app verifies your identity and only grants access to projects assigned to you."
                        if admin_storage_mode
                        else "Secure login with your Hugging Face account to access your validation workspace."
                    )
                )
                gr.LoginButton(
                    visible=False,
                    elem_id="bn-hidden-hf-oauth-route",
                )
                login_button = gr.Button(
                    "Sign in with Hugging Face",
                    variant="primary",
                    elem_classes=["bn-oauth-login-button"],
                )
            elif allow_username_login:
                gr.Markdown("Local development login")
                username_input = gr.Textbox(
                    label="Username",
                    placeholder="Enter your username",
                    lines=1,
                )
                login_button = gr.Button("Login", variant="primary", scale=1)
            error_message = gr.Markdown()
        with gr.Column(scale=1):
            gr.Markdown("")

    if enable_oauth_login and login_button is not None:
        login_button.click(
            fn=lambda: None,
            inputs=None,
            outputs=None,
            js="""
            () => {
                sessionStorage.setItem("birdnet_hf_login_intent", "1");
                window.parent?.postMessage({ type: "SET_SCROLLING", enabled: true }, "*");
                setTimeout(() => {
                    window.location.assign("/login/huggingface" + window.location.search);
                }, 100);
                return [];
            }
            """,
        )
    elif login_button is not None and username_input is not None:
        login_button.click(
            fn=lambda username: perform_login(
                auth_service,
                username,
                allow_username_login=True,
            ),
            inputs=[username_input],
            outputs=[session_output, error_message],
        )

    return session_output, error_message
