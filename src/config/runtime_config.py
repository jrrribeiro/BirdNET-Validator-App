from dataclasses import dataclass
import os
import tempfile
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    detection_seed_path: str | None
    validation_base_dir: str
    bootstrap_base_dir: str
    page_size: int
    projects_file_path: str | None
    user_access_file_path: str | None
    invites_file_path: str | None
    invite_ttl_hours: int
    enable_demo_bootstrap: bool
    invite_email_enabled: bool
    invite_email_sender: str
    invite_email_login_url: str
    smtp_host: str | None
    smtp_port: int
    smtp_username: str | None
    smtp_password: str | None
    smtp_use_tls: bool
    smtp_use_ssl: bool = False

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        detection_seed_path = (os.getenv("BIRDNET_DETECTIONS_FILE") or "").strip() or None

        data_root = Path("/data") if Path("/data").exists() else (Path(tempfile.gettempdir()) / "birdnet-validator-data")

        bootstrap_base_dir = (
            os.getenv("BIRDNET_BOOTSTRAP_DIR")
            or str(data_root / "bootstrap")
        )

        validation_base_dir = (
            os.getenv("BIRDNET_VALIDATIONS_DIR")
            or str(data_root / "validations")
        )

        raw_page_size = (os.getenv("BIRDNET_PAGE_SIZE") or "").strip()
        page_size = 25
        if raw_page_size:
            try:
                parsed = int(raw_page_size)
                if parsed > 0:
                    page_size = parsed
            except ValueError:
                page_size = 25

        projects_file_path = (os.getenv("BIRDNET_PROJECTS_FILE") or "").strip() or None
        user_access_file_path = (os.getenv("BIRDNET_USER_ACCESS_FILE") or "").strip() or None
        invites_file_path = (os.getenv("BIRDNET_INVITES_FILE") or "").strip() or None
        raw_invite_ttl_hours = (os.getenv("BIRDNET_INVITE_TTL_HOURS") or "").strip()
        invite_ttl_hours = 72
        if raw_invite_ttl_hours:
            try:
                parsed = int(raw_invite_ttl_hours)
                if parsed > 0:
                    invite_ttl_hours = parsed
            except ValueError:
                invite_ttl_hours = 72

        raw_enable_demo_bootstrap = (os.getenv("BIRDNET_ENABLE_DEMO_BOOTSTRAP") or "").strip().lower()
        enable_demo_bootstrap = raw_enable_demo_bootstrap in {"1", "true", "yes", "on"}

        raw_invite_email_enabled = (os.getenv("BIRDNET_INVITE_EMAIL_ENABLED") or "").strip().lower()
        invite_email_enabled = raw_invite_email_enabled in {"1", "true", "yes", "on"}
        invite_email_sender = (os.getenv("BIRDNET_INVITE_EMAIL_SENDER") or "").strip()
        invite_email_login_url = (os.getenv("BIRDNET_INVITE_EMAIL_LOGIN_URL") or "").strip() or ""

        smtp_host = (os.getenv("BIRDNET_SMTP_HOST") or "").strip() or None

        raw_smtp_port = (os.getenv("BIRDNET_SMTP_PORT") or "").strip()
        smtp_port = 587
        if raw_smtp_port:
            try:
                parsed = int(raw_smtp_port)
                if parsed > 0:
                    smtp_port = parsed
            except ValueError:
                smtp_port = 587

        smtp_username = (os.getenv("BIRDNET_SMTP_USERNAME") or "").strip() or None
        smtp_password = (os.getenv("BIRDNET_SMTP_PASSWORD") or "").strip() or None
        raw_smtp_use_tls = (os.getenv("BIRDNET_SMTP_USE_TLS") or "").strip().lower()
        smtp_use_tls = raw_smtp_use_tls not in {"0", "false", "no", "off"}
        raw_smtp_use_ssl = (os.getenv("BIRDNET_SMTP_USE_SSL") or "").strip().lower()
        smtp_use_ssl = raw_smtp_use_ssl in {"1", "true", "yes", "on"}

        return cls(
            detection_seed_path=detection_seed_path,
            validation_base_dir=validation_base_dir,
            bootstrap_base_dir=bootstrap_base_dir,
            page_size=page_size,
            projects_file_path=projects_file_path,
            user_access_file_path=user_access_file_path,
            invites_file_path=invites_file_path,
            invite_ttl_hours=invite_ttl_hours,
            enable_demo_bootstrap=enable_demo_bootstrap,
            invite_email_enabled=invite_email_enabled,
            invite_email_sender=invite_email_sender,
            invite_email_login_url=invite_email_login_url,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_username=smtp_username,
            smtp_password=smtp_password,
            smtp_use_tls=smtp_use_tls,
            smtp_use_ssl=smtp_use_ssl,
        )
