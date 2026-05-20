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
