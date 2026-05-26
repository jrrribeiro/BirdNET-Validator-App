"""Gradio login page component for multi-project authorization."""

from typing import Tuple

import gradio as gr

from src.auth.auth_service import AuthService


def perform_login(
    auth_service: AuthService,
    username: str,
    hf_token: str,
    *,
    allow_username_login: bool = True,
) -> Tuple[str, str]:
    """Attempt login and return session ID plus a user-facing status."""
    if hf_token and hf_token.strip():
        session, message = auth_service.login_with_hf_token(hf_token)
        if session is None:
            return "", message
        return session.session_id, message

    if not allow_username_login:
        return (
            "",
            "Username-only login is disabled for this deployment. Sign in with a Hugging Face token so your identity can be verified.",
        )

    if not username or not username.strip():
        return "", "Please enter a username or provide a Hugging Face token"

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
    trusted_team_mode: bool = False,
) -> Tuple[gr.Textbox, gr.Textbox, gr.Button, gr.Markdown]:
    """Create a Gradio login page with username input and session tracking.

    Args:
        auth_service: AuthService instance for login validation

    Returns:
        Tuple of (username_input, session_output, login_button, error_message)
    """
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("")
        with gr.Column(scale=6):
            gr.Markdown("# BirdNET Validation Platform")
            gr.Markdown(
                (
                    "Use your own Hugging Face authorization to access private project validation workflows."
                    if trusted_team_mode
                    else "Sign in with your Hugging Face account to access project validation workflows."
                )
            )
            if auth_mode_label:
                gr.Markdown(auth_mode_label)

            if trusted_team_mode:
                gr.Markdown(
                    "### Personal token access\n"
                    "This is the validated access method for private trusted-team projects. "
                    "Use a token from your own account with permission to read the project's private dataset "
                    "and write to its private validation Bucket."
                )
            else:
                gr.Markdown(
                    "Manual token access is available for development or legacy access, "
                    "but cannot be used for the private collaborative state authorization test."
                )
            username_input = gr.Textbox(
                label="Username",
                placeholder="Enter your username" if allow_username_login else "Username login disabled in this deployment",
                lines=1,
                interactive=allow_username_login,
                visible=allow_username_login,
            )
            hf_token_input = gr.Textbox(
                label=(
                    "Personal Hugging Face Token"
                    if trusted_team_mode
                    else "Hugging Face Token (legacy/development only)"
                ),
                placeholder="hf_xxx...",
                type="password",
                lines=1,
            )

            error_message = gr.Markdown()

            login_button = gr.Button(
                "Enter with personal token" if trusted_team_mode else "Login",
                variant="primary",
                scale=1,
            )

            oauth_continue_button = None
            if enable_oauth_login:
                gr.Markdown(
                    (
                        "### Optional OAuth test\n"
                        "The hosted Space can sign you in with Hugging Face OAuth. "
                        "For this interim private-storage mode, complete the personal-token test above before "
                        "using real validation data because OAuth Bucket writes have not yet been proven."
                        if trusted_team_mode
                        else (
                            "**OAuth access required for private collaborative state.** "
                            "Complete both steps below; signing into Hugging Face alone does not open an app session."
                        )
                    )
                )
                gr.LoginButton("1. Sign in with Hugging Face", logout_value="Sign out ({})")
                oauth_continue_button = gr.Button(
                    "2. Enter workspace with OAuth authorization",
                    variant="secondary" if trusted_team_mode else "primary",
                )

            session_output = gr.Textbox(
                label="Session ID",
                interactive=False,
                visible=False,
            )
        with gr.Column(scale=1):
            gr.Markdown("")

    login_button.click(
        fn=lambda username, hf_token: perform_login(
            auth_service,
            username,
            hf_token,
            allow_username_login=allow_username_login,
        ),
        inputs=[username_input, hf_token_input],
        outputs=[session_output, error_message],
    )

    def handle_oauth_login(
        profile: gr.OAuthProfile | None,
        oauth_token: gr.OAuthToken | None,
    ) -> Tuple[str, str]:
        return perform_oauth_login(auth_service, profile, oauth_token)

    if oauth_continue_button is not None:
        oauth_continue_button.click(
            fn=handle_oauth_login,
            inputs=None,
            outputs=[session_output, error_message],
        )

    return username_input, session_output, login_button, error_message
