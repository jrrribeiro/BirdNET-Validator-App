from dataclasses import dataclass
import os
import tempfile
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    detection_seed_path: str | None
    validation_base_dir: str
    page_size: int
    projects_file_path: str | None
    user_access_file_path: str | None
    invites_file_path: str | None
    invite_ttl_hours: int
    enable_demo_bootstrap: bool

    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        detection_seed_path = (os.getenv("BIRDNET_DETECTIONS_FILE") or "").strip() or None

        validation_base_dir = (
            os.getenv("BIRDNET_VALIDATIONS_DIR")
            or str(Path(tempfile.gettempdir()) / "birdnet-validator-validations")
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

        return cls(
            detection_seed_path=detection_seed_path,
            validation_base_dir=validation_base_dir,
            page_size=page_size,
            projects_file_path=projects_file_path,
            user_access_file_path=user_access_file_path,
            invites_file_path=invites_file_path,
            invite_ttl_hours=invite_ttl_hours,
            enable_demo_bootstrap=enable_demo_bootstrap,
        )
