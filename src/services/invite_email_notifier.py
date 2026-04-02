"""Email delivery via EmailJS for project collaboration invites."""

from dataclasses import dataclass
from datetime import datetime
import json
from urllib import request


@dataclass(frozen=True)
class InviteEmailPayload:
    """Payload for EmailJS invite notification.
    
    At least one of invitee_username or invitee_email must be present.
    - Only username: internal app invite only (no email sent).
    - Only email: external email-based invite (no prior HF account required).
    - Both: dual invite (internal + email).
    """
    project_slug: str
    role: str
    invited_by: str
    expires_at: datetime
    login_url: str
    invitee_username: str | None = None
    invitee_email: str | None = None


class InviteEmailNotifier:
    """Abstract notifier interface for invite emails."""

    def send(self, payload: InviteEmailPayload) -> tuple[bool, str]:
        """Send an invite notification. Returns (success, message)."""
        raise NotImplementedError


class EmailJSInviteEmailNotifier(InviteEmailNotifier):
    """EmailJS-backed notifier for invite emails (only transport available)."""

    def __init__(
        self,
        sender_email: str,
        service_id: str,
        template_id: str,
        public_key: str,
        endpoint: str = "https://api.emailjs.com/api/v1.0/email/send",
        timeout_seconds: int = 20,
    ):
        self._sender_email = sender_email
        self._service_id = service_id
        self._template_id = template_id
        self._public_key = public_key
        self._endpoint = endpoint
        self._timeout_seconds = max(5, int(timeout_seconds))

    def send(self, payload: InviteEmailPayload) -> tuple[bool, str]:
        """Send invite email via EmailJS.
        
        If invitee_email is missing, returns success with message indicating internal-only invite.
        """
        if not payload.invitee_email:
            return True, f"✅ Internal invite created for {payload.invitee_username or 'pending'}"

        invitee_display = payload.invitee_username or payload.invitee_email
        template_params = {
            "from_name": self._sender_email,
            "to_email": payload.invitee_email,
            "invitee_username": payload.invitee_username or "(to be claimed)",
            "project_slug": payload.project_slug,
            "role": payload.role,
            "invited_by": payload.invited_by,
            "expires_at": payload.expires_at.isoformat(),
            "login_url": payload.login_url or "",
            "invite_link": payload.login_url or "",
        }

        req_payload = {
            "service_id": self._service_id,
            "template_id": self._template_id,
            "user_id": self._public_key,
            "template_params": template_params,
        }

        try:
            req = request.Request(
                self._endpoint,
                data=json.dumps(req_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=self._timeout_seconds) as response:
                status_code = int(getattr(response, "status", 0) or response.getcode())
                if 200 <= status_code < 300:
                    return True, f"✅ Invite sent to {payload.invitee_email}"
                return False, f"Internal invite created, but email send failed (status {status_code})"
        except Exception as exc:
            return False, f"Internal invite created, but email send failed: {exc}"
