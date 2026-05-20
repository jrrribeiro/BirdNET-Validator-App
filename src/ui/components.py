from html import escape


def section_header_html(kicker: str, title: str, subtitle: str, *, class_name: str = "bn-panel") -> str:
    return (
        f"<div class='{class_name}' style='margin-bottom:12px;'>"
        f"<div class='bn-brand-kicker'>{escape(kicker)}</div>"
        f"<div class='bn-section-title'>{escape(title)}</div>"
        f"<div class='bn-compact-note'>{escape(subtitle)}</div>"
        "</div>"
    )


def status_pill_html(label: str, value: str, tone: str = "") -> str:
    tone_class = f" bn-pill-{tone}" if tone else ""
    return f"<span class='bn-pill{tone_class}'><strong>{escape(label)}:</strong>&nbsp;{escape(value)}</span>"


def project_overview_html(project_rows: list[dict], authorized_projects: list[str], selected_project: str | None = None) -> str:
    if not authorized_projects:
        return (
            "<div class='bn-empty-state'>"
            "<div class='bn-empty-title'>No projects assigned</div>"
            "<div class='bn-compact-note'>Create a project in Admin or accept a pending invite to start validating.</div>"
            "</div>"
        )

    rows_by_slug = {str(row.get("project_slug", "")): row for row in project_rows}
    cards: list[str] = []
    for slug in authorized_projects:
        row = rows_by_slug.get(slug, {})
        name = str(row.get("name") or slug)
        dataset_repo = str(row.get("dataset_repo_id") or "dataset not set")
        visibility = str(row.get("visibility") or "collaborative")
        active = "active" if bool(row.get("active", True)) else "inactive"
        token = "token set" if bool(row.get("dataset_token_set", False)) else "public/default token"
        selected_class = " bn-project-card-selected" if selected_project == slug else ""
        cards.append(
            f"<div class='bn-project-card{selected_class}'>"
            f"<div class='bn-project-name'>{escape(name)}</div>"
            f"<div class='bn-project-slug'>{escape(slug)}</div>"
            f"<div class='bn-compact-note'>{escape(dataset_repo)}</div>"
            "<div class='bn-card-pills'>"
            f"{status_pill_html('visibility', visibility)}"
            f"{status_pill_html('status', active, 'ok' if active == 'active' else 'warn')}"
            f"{status_pill_html('access', token)}"
            "</div>"
            "</div>"
        )

    return "<div class='bn-project-grid'>" + "".join(cards) + "</div>"


def compact_metric_grid(items: list[tuple[str, str, str, str]]) -> str:
    cards = []
    for label, value, hint, tone in items:
        tone_class = f" bn-kpi-{tone}" if tone else ""
        cards.append(
            f"<div class='bn-kpi-card{tone_class}'>"
            f"<div class='bn-kpi-label'>{escape(label)}</div>"
            f"<div class='bn-kpi-value'>{escape(value)}</div>"
            f"<div class='bn-kpi-hint'>{escape(hint)}</div>"
            "</div>"
        )
    return "<div class='bn-kpi-grid'>" + "".join(cards) + "</div>"


def admin_overview_html(
    *,
    username: str | None,
    total_projects: int,
    admin_projects: int,
    validator_projects: int,
    pending_invites: int,
) -> str:
    if not username:
        return (
            "<div class='bn-empty-state'>"
            "<div class='bn-empty-title'>Admin workspace locked</div>"
            "<div class='bn-compact-note'>Login to manage projects, teams, invites, and dataset tokens.</div>"
            "</div>"
        )

    return compact_metric_grid(
        [
            ("User", username, "active session", "info"),
            ("Projects", str(total_projects), "registered in app", ""),
            ("Admin scope", str(admin_projects), "projects you manage", "positive" if admin_projects else "warning"),
            ("Validator scope", str(validator_projects), f"{pending_invites} pending invites", "info"),
        ]
    )


def project_context_html(project_row: list[object] | None, role: str | None = None) -> str:
    if not project_row:
        return (
            "<div class='bn-empty-state'>"
            "<div class='bn-empty-title'>No project selected</div>"
            "<div class='bn-compact-note'>Choose an authorized project before opening the validation workbench.</div>"
            "</div>"
        )

    def cell(index: int, default: str = "") -> str:
        if len(project_row) <= index:
            return default
        return str(project_row[index] or "").strip() or default

    slug = cell(0, "unknown-project")
    name = cell(1, slug)
    dataset_repo = cell(2, "dataset not set")
    visibility = cell(3, "collaborative")
    owner = cell(4, "unknown")
    token = "token set" if cell(5, "no").lower() == "yes" else "public/default token"
    active = cell(6, "yes")
    role_label = (role or "unknown").upper()
    role_tone = "ok" if role_label == "ADMIN" else "info"
    active_tone = "ok" if active.lower() == "yes" else "warn"

    return (
        "<div class='bn-project-context'>"
        "<div>"
        "<div class='bn-brand-kicker'>Selected project</div>"
        f"<div class='bn-project-context-title'>{escape(name)}</div>"
        f"<div class='bn-project-context-subtitle'>{escape(slug)} · {escape(dataset_repo)}</div>"
        "</div>"
        "<div class='bn-card-pills'>"
        f"{status_pill_html('role', role_label, role_tone)}"
        f"{status_pill_html('visibility', visibility)}"
        f"{status_pill_html('owner', owner)}"
        f"{status_pill_html('token', token)}"
        f"{status_pill_html('active', active, active_tone)}"
        "</div>"
        "</div>"
    )


def invite_panel_html(invite_count: int) -> str:
    if invite_count <= 0:
        return inline_hint_html("No pending invites for this user.", "info")
    return (
        "<div class='bn-invite-callout'>"
        f"<div class='bn-empty-title'>{invite_count} pending invite{'s' if invite_count != 1 else ''}</div>"
        "<div class='bn-compact-note'>Review and accept only the project access you expect to use.</div>"
        "</div>"
    )


def coverage_bars_html(rows: list[list[object]], *, limit: int = 12) -> str:
    if not rows:
        return (
            "<div class='bn-coverage-panel'>"
            "<div class='bn-queue-preview-head'><span>Coverage by species</span><span>0 species</span></div>"
            "<div class='bn-empty-mini'>Refresh the dashboard after selecting a project.</div>"
            "</div>"
        )

    cards: list[str] = []
    for row in rows[:limit]:
        species = str(row[0] if len(row) > 0 else "Unknown species")
        total = int(row[1] or 0) if len(row) > 1 else 0
        validated = int(row[2] or 0) if len(row) > 2 else 0
        remaining = int(row[3] or 0) if len(row) > 3 else max(0, total - validated)
        try:
            pct = float(row[4] if len(row) > 4 else 0.0)
        except Exception:
            pct = 0.0
        pct = max(0.0, min(100.0, pct))
        cards.append(
            "<div class='bn-coverage-row'>"
            "<div class='bn-coverage-row-head'>"
            f"<span>{escape(species)}</span>"
            f"<strong>{pct:.1f}%</strong>"
            "</div>"
            "<div class='bn-coverage-track'>"
            f"<div class='bn-coverage-fill' style='width:{pct:.1f}%;'></div>"
            "</div>"
            f"<div class='bn-coverage-meta'>{validated} validated · {remaining} remaining · {total} total</div>"
            "</div>"
        )

    remaining_species = max(0, len(rows) - limit)
    footer = f"<div class='bn-queue-footnote'>+{remaining_species} species in the table below</div>" if remaining_species else ""
    return (
        "<div class='bn-coverage-panel'>"
        "<div class='bn-queue-preview-head'>"
        f"<span>Coverage by species</span><span>{len(rows)} species</span>"
        "</div>"
        + "".join(cards)
        + footer
        + "</div>"
    )


def settings_health_html(items: list[tuple[str, str, str]]) -> str:
    cards = []
    for label, value, tone in items:
        tone_class = f" bn-pill-{tone}" if tone else ""
        cards.append(
            "<div class='bn-health-row'>"
            f"<div><div class='bn-health-label'>{escape(label)}</div><div class='bn-health-value'>{escape(value)}</div></div>"
            f"<span class='bn-pill{tone_class}'>{escape(tone or 'info')}</span>"
            "</div>"
        )
    return "<div class='bn-health-panel'>" + "".join(cards) + "</div>"


def inline_hint_html(text: str, tone: str = "info") -> str:
    tone_class = f" bn-hint-{tone}" if tone else ""
    return f"<div class='bn-inline-hint{tone_class}'>{escape(text)}</div>"


def selected_segment_html(row: list[object] | None, selected_index: int | None = None, total_rows: int | None = None) -> str:
    if not row:
        return (
            "<div class='bn-selected-card'>"
            "<div class='bn-brand-kicker'>Selected segment</div>"
            "<div class='bn-empty-title'>No segment loaded</div>"
            "<div class='bn-compact-note'>Choose a species and apply filters to load the validation queue.</div>"
            "</div>"
        )

    def cell(index: int, default: str = "") -> str:
        if len(row) <= index:
            return default
        value = str(row[index] or "").strip()
        if index == 2 and value.startswith("▶ "):
            value = value[2:].strip()
        return value or default

    detection_key = cell(0, "unknown")
    audio_id = cell(1, "unknown audio")
    species = cell(2, "Unknown species")
    confidence = cell(3, "0")
    start_time = cell(4, "0")
    end_time = cell(5, "0")
    status = cell(6, "pending")
    version = cell(7, "0")
    conflict = cell(8, "")
    position = ""
    if selected_index is not None and total_rows:
        position = f"{int(selected_index) + 1}/{int(total_rows)}"

    conflict_badge = "<span class='bn-pill bn-pill-warn'>conflict</span>" if conflict else ""
    return (
        "<div class='bn-selected-card'>"
        "<div class='bn-selected-topline'>"
        "<div>"
        "<div class='bn-brand-kicker'>Selected segment</div>"
        f"<div class='bn-selected-species'>{escape(species)}</div>"
        "</div>"
        f"<span class='bn-pill bn-pill-info'>{escape(position or 'queue')}</span>"
        "</div>"
        "<div class='bn-selected-meta'>"
        f"<span><strong>Confidence</strong>{escape(confidence)}</span>"
        f"<span><strong>Time</strong>{escape(start_time)}-{escape(end_time)}s</span>"
        f"<span><strong>Status</strong>{escape(status)}</span>"
        f"<span><strong>Version</strong>{escape(version)}</span>"
        "</div>"
        f"<div class='bn-selected-audio'>{escape(audio_id)}</div>"
        "<div class='bn-card-pills'>"
        f"<span class='bn-pill'>key: {escape(detection_key)}</span>"
        f"{conflict_badge}"
        "</div>"
        "</div>"
    )


def validation_queue_html(rows: object, selected_index: int | None = None, *, limit: int = 8) -> str:
    if hasattr(rows, "values"):
        normalized_rows = [list(item) for item in rows.values.tolist()]
    else:
        normalized_rows = [list(item) for item in rows] if rows else []

    if not normalized_rows:
        return (
            "<div class='bn-queue-preview'>"
            "<div class='bn-queue-preview-head'>"
            "<span>Queue preview</span><span>0 loaded</span>"
            "</div>"
            "<div class='bn-empty-mini'>Select a species and apply filters to load segments.</div>"
            "</div>"
        )

    safe_selected = max(0, min(int(selected_index or 0), len(normalized_rows) - 1))
    cards: list[str] = []
    for idx, row in enumerate(normalized_rows[:limit]):
        def cell(index: int, default: str = "") -> str:
            if len(row) <= index:
                return default
            value = str(row[index] or "").strip()
            if index == 2 and value.startswith("▶ "):
                value = value[2:].strip()
            return value or default

        species = cell(2, "Unknown species")
        confidence = cell(3, "0")
        status = cell(6, "pending")
        audio_id = cell(1, "unknown audio")
        selected_class = " bn-queue-card-selected" if idx == safe_selected else ""
        cards.append(
            f"<div class='bn-queue-card{selected_class}'>"
            f"<div class='bn-queue-card-index'>{idx + 1}</div>"
            "<div class='bn-queue-card-main'>"
            f"<div class='bn-queue-card-species'>{escape(species)}</div>"
            f"<div class='bn-queue-card-audio'>{escape(audio_id)}</div>"
            "</div>"
            "<div class='bn-queue-card-meta'>"
            f"<span>{escape(confidence)}</span>"
            f"<span>{escape(status)}</span>"
            "</div>"
            "</div>"
        )

    remaining = max(0, len(normalized_rows) - limit)
    remaining_text = f"+{remaining} more on this page" if remaining else "all loaded rows shown"
    return (
        "<div class='bn-queue-preview'>"
        "<div class='bn-queue-preview-head'>"
        f"<span>Queue preview</span><span>{safe_selected + 1}/{len(normalized_rows)}</span>"
        "</div>"
        + "".join(cards)
        + f"<div class='bn-queue-footnote'>{escape(remaining_text)}</div>"
        "</div>"
    )
