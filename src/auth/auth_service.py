"""AuthService: User authentication, session management, and project ACL."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Dict, List, Optional
import uuid

from huggingface_hub import HfApi

from src.domain.models import Role, User


@dataclass
class UserProjectAccess:
    """Maps a user to projects with specific roles."""

    username: str
    project_slugs: Dict[str, Role]  # Maps project_slug -> role (admin or validator)
    is_active: bool = True


@dataclass
class Session:
    """Represents an authenticated user session."""

    session_id: str
    username: str
    role: Role
    authorized_projects: List[str]  # Projects this user can access
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    hf_token: str | None = None

    def is_expired(self) -> bool:
        """Check if session has expired."""
        return datetime.now(UTC) > self.expires_at

    def update_activity(self, ttl_minutes: int = 120) -> None:
        """Update last activity timestamp and extend expiration."""
        self.last_activity = datetime.now(UTC)
        self.expires_at = datetime.now(UTC) + timedelta(minutes=ttl_minutes)


@dataclass
class ProjectInvite:
    username: str
    project_slug: str
    role: Role
    invited_by: str
    created_at: datetime
    expires_at: datetime

    def is_expired(self) -> bool:
        return datetime.now(UTC) > self.expires_at


class AuthService:
    """Manages user authentication, sessions, and project access control."""

    def __init__(self, session_ttl_minutes: int = 120, invite_ttl_hours: int = 72):
        """Initialize AuthService with session TTL.

        Args:
            session_ttl_minutes: Session time-to-live in minutes (default: 2 hours)
        """
        self.session_ttl_minutes = session_ttl_minutes
        self.invite_ttl_hours = max(1, int(invite_ttl_hours))
        self._sessions: Dict[str, Session] = {}  # session_id -> Session
        self._user_access: Dict[str, UserProjectAccess] = {}  # username -> UserProjectAccess
        self._pending_invites: Dict[str, Dict[str, ProjectInvite]] = {}

    def _prune_expired_invites(self) -> None:
        expired_users: list[str] = []
        for username, invites_by_project in self._pending_invites.items():
            expired_projects = [project_slug for project_slug, invite in invites_by_project.items() if invite.is_expired()]
            for project_slug in expired_projects:
                del invites_by_project[project_slug]
            if not invites_by_project:
                expired_users.append(username)

        for username in expired_users:
            self._pending_invites.pop(username, None)

    def register_user_project_access(
        self, username: str, project_access: Dict[str, Role]
    ) -> None:
        """Register or update a user's access to projects.

        Args:
            username: User name
            project_access: Dict mapping project_slug to Role (admin or validator)
        """
        self._user_access[username] = UserProjectAccess(
            username=username,
            project_slugs=project_access,
            is_active=True,
        )

    def list_usernames(self, include_inactive: bool = False) -> List[str]:
        """List registered usernames.

        Args:
            include_inactive: Include disabled users when True

        Returns:
            Sorted list of usernames
        """
        usernames = []
        for username, access in self._user_access.items():
            if include_inactive or access.is_active:
                usernames.append(username)
        return sorted(usernames)

    def upsert_user_project_role(self, username: str, project_slug: str, role: Role) -> None:
        """Create/update user assignment for a project.

        Args:
            username: User name
            project_slug: Project slug
            role: Role to apply for this project
        """
        access = self._user_access.get(username)
        if access is None:
            access = UserProjectAccess(username=username, project_slugs={}, is_active=True)
            self._user_access[username] = access

        access.project_slugs[project_slug] = role

    def remove_user_project_role(self, username: str, project_slug: str) -> bool:
        """Remove a user assignment from a project.

        Args:
            username: User name
            project_slug: Project slug

        Returns:
            True when assignment existed and was removed
        """
        access = self._user_access.get(username)
        if access is None:
            return False

        if project_slug not in access.project_slugs:
            return False

        del access.project_slugs[project_slug]
        return True

    def login(self, username: str) -> Optional[Session]:
        """Authenticate a user and create a new session.

        Args:
            username: User name to authenticate

        Returns:
            Session if user exists and is active, None otherwise
        """
        return self.login_internal(username=username)

    def login_internal(
        self,
        username: str,
        hf_token: str | None = None,
        auto_promote_to_admin: bool = False,
    ) -> Optional[Session]:
        if username not in self._user_access:
            if not auto_promote_to_admin:
                self._prune_expired_invites()
                pending_for_user = self._pending_invites.get(username, {})
                if not pending_for_user:
                    return None
            self._user_access[username] = UserProjectAccess(
                username=username,
                project_slugs={},
                is_active=True,
            )

        access = self._user_access[username]
        if not access.is_active:
            return None

        # Determine highest role across all projects (admin > validator)
        role = Role.admin if auto_promote_to_admin else Role.validator
        if role != Role.admin:
            for proj_role in access.project_slugs.values():
                if proj_role == Role.admin:
                    role = Role.admin
                    break

        session_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        session = Session(
            session_id=session_id,
            username=username,
            role=role,
            authorized_projects=list(access.project_slugs.keys()),
            created_at=now,
            last_activity=now,
            expires_at=now + timedelta(minutes=self.session_ttl_minutes),
            hf_token=hf_token,
        )

        self._sessions[session_id] = session
        return session

    def login_with_hf_token(self, token: str) -> tuple[Optional[Session], str]:
        """Authenticate using Hugging Face personal token and resolve username via whoami."""
        token_value = (token or "").strip()
        if not token_value:
            return None, "❌ Please provide a Hugging Face token"

        try:
            whoami = HfApi().whoami(token=token_value)
        except Exception:
            return None, "❌ Invalid Hugging Face token or network error"

        username = str(whoami.get("name") or "").strip()
        if not username:
            return None, "❌ Unable to resolve Hugging Face username from token"

        is_first_user = len(self._user_access) == 0
        session = self.login_internal(
            username=username,
            hf_token=token_value,
            auto_promote_to_admin=is_first_user,
        )
        if session is None:
            return None, f"❌ User '{username}' is not invited to any project yet"

        if is_first_user:
            return session, f"✅ Welcome, {username}! (Admin)"
        if session.role == Role.admin:
            return session, f"✅ Welcome, {username}! (Admin)"
        return session, f"✅ Welcome, {username}! (Validator)"

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve an active session by ID.

        Args:
            session_id: Session ID

        Returns:
            Session if valid and not expired, None otherwise
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        if session.is_expired():
            self._sessions.pop(session_id, None)
            return None

        session.update_activity(self.session_ttl_minutes)
        return session

    def logout(self, session_id: str) -> None:
        """Delete a session.

        Args:
            session_id: Session ID to invalidate
        """
        self._sessions.pop(session_id, None)

    def is_user_authorized_for_project(self, username: str, project_slug: str) -> bool:
        """Check if a user has access to a specific project.

        Args:
            username: User name
            project_slug: Project slug

        Returns:
            True if user has access (admin or validator), False otherwise
        """
        access = self._user_access.get(username)
        if access is None or not access.is_active:
            return False

        return project_slug in access.project_slugs

    def get_user_role_for_project(self, username: str, project_slug: str) -> Optional[Role]:
        """Get the specific role a user has for a project.

        Args:
            username: User name
            project_slug: Project slug

        Returns:
            Role if user has access, None otherwise
        """
        access = self._user_access.get(username)
        if access is None or not access.is_active:
            return None

        return access.project_slugs.get(project_slug)

    def list_user_projects(self, username: str) -> List[str]:
        """List all projects a user has access to.

        Args:
            username: User name

        Returns:
            List of project slugs
        """
        access = self._user_access.get(username)
        if access is None or not access.is_active:
            return []

        return list(access.project_slugs.keys())

    def cleanup_expired_sessions(self) -> None:
        """Remove all expired sessions (maintenance cleanup)."""
        now = datetime.now(UTC)
        expired = [
            sid for sid, session in self._sessions.items() if session.is_expired()
        ]
        for sid in expired:
            self._sessions.pop(sid)

    def refresh_session_authorizations(self, session_id: str) -> Optional[Session]:
        """Refresh a session with latest role and project assignments."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if session.is_expired():
            self._sessions.pop(session_id, None)
            return None

        access = self._user_access.get(session.username)
        if access is None or not access.is_active:
            self._sessions.pop(session_id, None)
            return None

        projects = list(access.project_slugs.keys())
        session.authorized_projects = projects
        session.role = Role.admin if any(role == Role.admin for role in access.project_slugs.values()) else Role.validator
        session.update_activity(self.session_ttl_minutes)
        return session

    def set_user_active(self, username: str, active: bool) -> None:
        """Enable or disable a user account.

        Args:
            username: User name
            active: Whether the user should be active
        """
        if username in self._user_access:
            self._user_access[username].is_active = active

    def create_project_invite(
        self,
        username: str,
        project_slug: str,
        role: Role,
        invited_by: str,
    ) -> tuple[bool, str]:
        invited_username = (username or "").strip()
        if not invited_username:
            return False, "Username is required"

        self._prune_expired_invites()

        user_invites = self._pending_invites.setdefault(invited_username, {})
        if project_slug in user_invites:
            return False, f"Invite for {invited_username} to {project_slug} already exists"

        now = datetime.now(UTC)
        expires_at = now + timedelta(hours=self.invite_ttl_hours)

        user_invites[project_slug] = ProjectInvite(
            username=invited_username,
            project_slug=project_slug,
            role=role,
            invited_by=invited_by,
            created_at=now,
            expires_at=expires_at,
        )
        return True, f"✅ Invite sent to {invited_username} for {project_slug} as {role.value} (expires {expires_at.isoformat()})"

    def list_pending_invites(self, username: str) -> List[ProjectInvite]:
        self._prune_expired_invites()
        invites = list(self._pending_invites.get(username, {}).values())
        invites.sort(key=lambda item: item.created_at)
        return invites

    def list_all_pending_invites(self) -> List[ProjectInvite]:
        self._prune_expired_invites()
        invites: list[ProjectInvite] = []
        for invites_by_project in self._pending_invites.values():
            invites.extend(invites_by_project.values())
        invites.sort(key=lambda item: item.created_at)
        return invites

    def accept_project_invite(self, username: str, project_slug: str) -> tuple[bool, str]:
        self._prune_expired_invites()
        pending = self._pending_invites.get(username, {})
        invite = pending.get(project_slug)
        if invite is None:
            return False, f"Invite not found for {username} in {project_slug}"

        self.upsert_user_project_role(username, project_slug, invite.role)
        del pending[project_slug]
        if not pending:
            self._pending_invites.pop(username, None)
        return True, f"✅ Invite accepted for {project_slug} as {invite.role.value}"

    def reject_project_invite(self, username: str, project_slug: str) -> tuple[bool, str]:
        self._prune_expired_invites()
        pending = self._pending_invites.get(username, {})
        invite = pending.get(project_slug)
        if invite is None:
            return False, f"Invite not found for {username} in {project_slug}"

        del pending[project_slug]
        if not pending:
            self._pending_invites.pop(username, None)
        return True, f"✅ Invite rejected for {project_slug}"

    def accept_all_project_invites(self, username: str) -> tuple[int, int, str]:
        self._prune_expired_invites()
        pending = self._pending_invites.get(username, {})
        if not pending:
            return 0, 0, "No pending invites"

        accepted = 0
        failed = 0
        for project_slug in list(pending.keys()):
            ok, _ = self.accept_project_invite(username, project_slug)
            if ok:
                accepted += 1
            else:
                failed += 1
        return accepted, failed, f"Accepted {accepted} invites" + (f", {failed} failed" if failed else "")

    def revoke_project_invite(self, username: str, project_slug: str) -> tuple[bool, str]:
        self._prune_expired_invites()
        pending = self._pending_invites.get(username, {})
        if project_slug not in pending:
            return False, f"Invite not found for {username} in {project_slug}"
        del pending[project_slug]
        if not pending:
            self._pending_invites.pop(username, None)
        return True, f"✅ Invite revoked for {username} in {project_slug}"

    def export_user_access_map(self, include_inactive: bool = False) -> Dict[str, Dict[str, str]]:
        """Export user/project role mapping for persistence."""
        exported: Dict[str, Dict[str, str]] = {}
        for username in self.list_usernames(include_inactive=include_inactive):
            access = self._user_access.get(username)
            if access is None:
                continue
            exported[username] = {project_slug: role.value for project_slug, role in access.project_slugs.items()}
        return exported

    def export_pending_invites_map(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        self._prune_expired_invites()
        payload: Dict[str, Dict[str, Dict[str, str]]] = {}
        for username, invites_by_project in self._pending_invites.items():
            payload[username] = {}
            for project_slug, invite in invites_by_project.items():
                payload[username][project_slug] = {
                    "role": invite.role.value,
                    "invited_by": invite.invited_by,
                    "created_at": invite.created_at.isoformat(),
                    "expires_at": invite.expires_at.isoformat(),
                }
        return payload

    def load_pending_invites_map(self, payload: Dict[str, Dict[str, Dict[str, str]]]) -> None:
        self._pending_invites = {}
        for username, invites_by_project in (payload or {}).items():
            if not isinstance(invites_by_project, dict):
                continue
            user_invites: Dict[str, ProjectInvite] = {}
            for project_slug, invite_payload in invites_by_project.items():
                if not isinstance(invite_payload, dict):
                    continue
                role_text = str(invite_payload.get("role", "")).strip().lower()
                if role_text not in {"admin", "validator"}:
                    continue
                invited_by = str(invite_payload.get("invited_by", "")).strip() or "admin"
                created_at_raw = str(invite_payload.get("created_at", "")).strip()
                expires_at_raw = str(invite_payload.get("expires_at", "")).strip()
                try:
                    created_at = datetime.fromisoformat(created_at_raw) if created_at_raw else datetime.now(UTC)
                except Exception:
                    created_at = datetime.now(UTC)
                try:
                    expires_at = datetime.fromisoformat(expires_at_raw) if expires_at_raw else (created_at + timedelta(hours=self.invite_ttl_hours))
                except Exception:
                    expires_at = created_at + timedelta(hours=self.invite_ttl_hours)
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=UTC)
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=UTC)
                user_invites[str(project_slug)] = ProjectInvite(
                    username=str(username),
                    project_slug=str(project_slug),
                    role=Role(role_text),
                    invited_by=invited_by,
                    created_at=created_at,
                    expires_at=expires_at,
                )

            if user_invites:
                self._pending_invites[str(username)] = user_invites

        self._prune_expired_invites()

    def revoke_all_invites_for_project(self, project_slug: str) -> int:
        """Remove all pending invites for a project.

        Returns:
            Number of revoked invites
        """
        self._prune_expired_invites()
        revoked = 0
        empty_users: list[str] = []
        for username, invites_by_project in self._pending_invites.items():
            if project_slug in invites_by_project:
                del invites_by_project[project_slug]
                revoked += 1
            if not invites_by_project:
                empty_users.append(username)

        for username in empty_users:
            self._pending_invites.pop(username, None)

        return revoked

    def remove_project_from_all_users(self, project_slug: str) -> int:
        """Remove a project assignment from all users and refresh active sessions.

        Returns:
            Number of user assignments removed
        """
        removed = 0
        for access in self._user_access.values():
            if project_slug in access.project_slugs:
                del access.project_slugs[project_slug]
                removed += 1

        for session in self._sessions.values():
            if project_slug in session.authorized_projects:
                session.authorized_projects = [slug for slug in session.authorized_projects if slug != project_slug]

                access = self._user_access.get(session.username)
                if access is None or not access.is_active:
                    continue
                session.role = (
                    Role.admin
                    if any(role == Role.admin for role in access.project_slugs.values())
                    else Role.validator
                )

        return removed
