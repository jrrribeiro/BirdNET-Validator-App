from src.ui.components import (
    admin_overview_html,
    coverage_bars_html,
    paged_activity_html,
    project_overview_html,
    project_context_html,
    validation_queue_html,
)


def test_validation_queue_html_highlights_selected_row_and_escapes_values():
    html = validation_queue_html(
        [
            ["key-1", "audio-one.wav", "Species <one>", "0.91", "0", "3", "pending"],
            ["key-2", "audio-two.wav", "Species two", "0.82", "4", "7", "positive"],
        ],
        selected_index=1,
    )

    assert "1/2" not in html
    assert "2/2" in html
    assert "bn-queue-card-selected" in html
    assert "Species &lt;one&gt;" in html
    assert "audio-two.wav" in html


def test_coverage_bars_html_renders_species_progress():
    html = coverage_bars_html(
        [
            ["Tinamus major", 100, 25, 75, 25.0],
            ["Attila spadiceus", 10, 10, 0, 100.0],
        ]
    )

    assert "Coverage by species" in html
    assert "Tinamus major" in html
    assert "25.0%" in html
    assert "width:100.0%;" in html


def test_paged_activity_html_limits_rows_and_reports_page():
    html = paged_activity_html(
        "Validator activity",
        ["Validator", "Validations"],
        [[f"validator-{index}", index] for index in range(12)],
        page=2,
        page_size=10,
    )

    assert "Validator activity" in html
    assert "validator-10" in html
    assert "validator-0" not in html
    assert "Page 2/2" in html


def test_project_context_html_renders_project_metadata():
    html = project_context_html(
        ["ppbio", "PPBIO", "jrrribeiro/ppbio", "private", "jonathan", "yes", "yes"],
        role="admin",
    )

    assert "PPBIO" in html
    assert "jrrribeiro/ppbio" in html
    assert "ADMIN" in html
    assert "token set" in html


def test_project_overview_html_accepts_table_rows():
    html = project_overview_html(
        [["ppbio", "PPBIO", "jrrribeiro/ppbio", "collaborative", "jonathan", "yes", "yes"]],
        ["ppbio"],
        "ppbio",
    )

    assert "PPBIO" in html
    assert "jrrribeiro/ppbio" in html
    assert "token set" in html
    assert "bn-project-card-selected" in html


def test_admin_overview_html_handles_locked_and_authenticated_states():
    locked = admin_overview_html(
        username=None,
        total_projects=3,
        admin_projects=0,
        validator_projects=0,
        pending_invites=0,
    )
    active = admin_overview_html(
        username="validator",
        total_projects=3,
        admin_projects=1,
        validator_projects=2,
        pending_invites=4,
    )

    assert "Admin workspace locked" in locked
    assert "validator" in active
    assert "4 pending invites" in active
