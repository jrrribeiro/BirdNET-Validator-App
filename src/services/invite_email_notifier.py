"""Email delivery adapter for project collaboration invites."""

from dataclasses import dataclass
from datetime import datetime
import json
import smtplib
from email.message import EmailMessage
from urllib import request


@dataclass(frozen=True)
class InviteEmailPayload:
    invitee_username: str
    invitee_email: str
    project_slug: str
    role: str
    invited_by: str
    expires_at: datetime
    login_url: str


class InviteEmailNotifier:
    """Abstract notifier used by admin flow to send invite emails."""

    def send(self, payload: InviteEmailPayload) -> tuple[bool, str]:
        raise NotImplementedError


class NoopInviteEmailNotifier(InviteEmailNotifier):
    """Fallback notifier when SMTP is not configured."""

    def send(self, payload: InviteEmailPayload) -> tuple[bool, str]:
        return False, "Invite email not sent: email notifications are disabled"


class HttpInviteEmailNotifier(InviteEmailNotifier):
    """HTTP webhook-based notifier for invite emails."""

    def __init__(
        self,
        sender_email: str,
        endpoint: str,
        api_key: str,
        timeout_seconds: int = 20,
    ):
        self._sender_email = sender_email
        self._endpoint = endpoint
        self._api_key = api_key
        self._timeout_seconds = max(5, int(timeout_seconds))

    def send(self, payload: InviteEmailPayload) -> tuple[bool, str]:
        body_text = (
            f"Hello {payload.invitee_username},\n\n"
            f"You were invited by {payload.invited_by} to collaborate on project '{payload.project_slug}' "
            f"with role '{payload.role}'.\n"
            f"Invite expires at: {payload.expires_at.isoformat()}\n\n"
            "How to join:\n"
            "1) Open Hugging Face and create an Access Token in Settings -> Access Tokens.\n"
            "2) Open BirdNET Validator login.\n"
            "3) Login using your Hugging Face token.\n"
            "4) Go to 'Select Project' and accept your invite.\n\n"
            f"Login URL: {payload.login_url or '(configured by admin)'}\n\n"
            "If you were not expecting this invitation, ignore this email.\n"
        )

        req_payload = {
            "from": self._sender_email,
            "to": [payload.invitee_email],
            "subject": f"BirdNET project invite: {payload.project_slug}",
            "text": body_text,
            "metadata": {
                "project_slug": payload.project_slug,
                "invitee_username": payload.invitee_username,
                "role": payload.role,
            },
        }

        try:
            req = request.Request(
                self._endpoint,
                data=json.dumps(req_payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=self._timeout_seconds) as response:
                status_code = int(getattr(response, "status", 0) or response.getcode())
                if 200 <= status_code < 300:
                    return True, f"Invite email sent to {payload.invitee_email}"
                return False, f"Invite created, but HTTP email API returned status {status_code}"
        except Exception as exc:
            return False, f"Invite created, but HTTP email delivery failed: {exc}"


class FallbackInviteEmailNotifier(InviteEmailNotifier):
    """Tries a primary notifier first, then falls back to a secondary notifier."""

    def __init__(self, primary: InviteEmailNotifier, secondary: InviteEmailNotifier):
        self._primary = primary
        self._secondary = secondary

    def send(self, payload: InviteEmailPayload) -> tuple[bool, str]:
        ok, status = self._primary.send(payload)
        if ok:
            return ok, status

        fallback_ok, fallback_status = self._secondary.send(payload)
        if fallback_ok:
            return True, f"{status} | fallback: {fallback_status}"
        return False, f"{status} | fallback: {fallback_status}"


class SmtpInviteEmailNotifier(InviteEmailNotifier):
    """SMTP-based email notifier for project invites."""

    def __init__(
        self,
        sender_email: str,
        smtp_host: str,
        smtp_port: int,
        smtp_username: str | None = None,
        smtp_password: str | None = None,
        smtp_use_tls: bool = True,
        smtp_use_ssl: bool = False,
    ):
        self._sender_email = sender_email
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._smtp_username = smtp_username
        self._smtp_password = smtp_password
        self._smtp_use_tls = smtp_use_tls
        self._smtp_use_ssl = smtp_use_ssl

    def send(self, payload: InviteEmailPayload) -> tuple[bool, str]:
        message = EmailMessage()
        message["From"] = self._sender_email
        message["To"] = payload.invitee_email
        message["Subject"] = f"BirdNET project invite: {payload.project_slug}"

        body = (
            f"Hello {payload.invitee_username},\n\n"
            f"You were invited by {payload.invited_by} to collaborate on project '{payload.project_slug}' "
            f"with role '{payload.role}'.\n"
            f"Invite expires at: {payload.expires_at.isoformat()}\n\n"
            "How to join:\n"
            "1) Open Hugging Face and create an Access Token in Settings -> Access Tokens.\n"
            "2) Open BirdNET Validator login.\n"
            "3) Login using your Hugging Face token.\n"
            "4) Go to 'Select Project' and accept your invite.\n\n"
            f"Login URL: {payload.login_url or '(configured by admin)'}\n\n"
            "If you were not expecting this invitation, ignore this email.\n"
        )
        message.set_content(body)

        try:
            if self._smtp_use_ssl:
                with smtplib.SMTP_SSL(self._smtp_host, self._smtp_port, timeout=20) as server:
                    if self._smtp_username and self._smtp_password:
                        server.login(self._smtp_username, self._smtp_password)
                    server.send_message(message)
            else:
                with smtplib.SMTP(self._smtp_host, self._smtp_port, timeout=20) as server:
                    if self._smtp_use_tls:
                        server.starttls()
                    if self._smtp_username and self._smtp_password:
                        server.login(self._smtp_username, self._smtp_password)
                    server.send_message(message)
            return True, f"Invite email sent to {payload.invitee_email}"
        except Exception as exc:
            return False, f"Invite created, but email delivery failed: {exc}"
