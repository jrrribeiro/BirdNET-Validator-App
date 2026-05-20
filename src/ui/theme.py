APP_CSS = """
:root {
  --bn-bg: #f5f7fb;
  --bn-panel: #ffffff;
  --bn-panel-soft: #f9fafb;
  --bn-border: #d9e0ea;
  --bn-border-strong: #c4cfdd;
  --bn-text: #142033;
  --bn-muted: #667085;
  --bn-primary: #1f6f8b;
  --bn-primary-dark: #14526a;
  --bn-positive: #0f7a4f;
  --bn-negative: #b42318;
  --bn-warning: #b54708;
  --bn-info: #255db3;
  --bn-shadow: 0 8px 24px rgba(20, 32, 51, 0.07);
}

body,
.gradio-container {
  background: var(--bn-bg) !important;
  color: var(--bn-text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.gradio-container {
  max-width: 1540px !important;
}

.bn-shell {
  padding-top: 8px;
}

.bn-app-header {
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: center;
  padding: 18px 22px;
  margin: 0 0 16px 0;
  background: linear-gradient(135deg, #ffffff 0%, #eef6f8 100%);
  border: 1px solid var(--bn-border);
  border-radius: 8px;
  box-shadow: var(--bn-shadow);
}

.bn-brand-kicker {
  color: var(--bn-primary-dark);
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0;
  text-transform: uppercase;
}

.bn-brand-title {
  margin-top: 2px;
  font-size: 26px;
  line-height: 1.15;
  font-weight: 780;
  color: var(--bn-text);
}

.bn-section-title {
  margin-top: 2px;
  font-size: 22px;
  line-height: 1.18;
  font-weight: 780;
  color: var(--bn-text);
}

.bn-brand-subtitle {
  margin-top: 4px;
  color: var(--bn-muted);
  font-size: 14px;
}

.bn-header-status {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.bn-pill {
  display: inline-flex;
  align-items: center;
  min-height: 30px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid var(--bn-border);
  background: rgba(255, 255, 255, 0.82);
  color: var(--bn-text);
  font-size: 12px;
  font-weight: 650;
  white-space: nowrap;
}

.bn-pill-ok {
  border-color: #b7dfc9;
  background: #ecfdf3;
  color: var(--bn-positive);
}

.bn-pill-info {
  border-color: #bfdbfe;
  background: #eff6ff;
  color: var(--bn-info);
}

.bn-pill-warn {
  border-color: #fed7aa;
  background: #fff7ed;
  color: var(--bn-warning);
}

.bn-tabs .tab-nav,
.tabs .tab-nav {
  gap: 4px !important;
  border-bottom: 1px solid var(--bn-border) !important;
}

.bn-tabs button,
.tabs button {
  border-radius: 6px 6px 0 0 !important;
  font-weight: 650 !important;
}

.bn-panel {
  background: var(--bn-panel);
  border: 1px solid var(--bn-border);
  border-radius: 8px;
  box-shadow: var(--bn-shadow);
  padding: 14px;
}

.bn-panel-soft {
  background: var(--bn-panel-soft);
  border: 1px solid var(--bn-border);
  border-radius: 8px;
  padding: 12px;
}

.bn-empty-state {
  padding: 18px;
  border: 1px dashed var(--bn-border-strong);
  background: #ffffff;
  border-radius: 8px;
}

.bn-empty-title {
  color: var(--bn-text);
  font-size: 16px;
  font-weight: 760;
  margin-bottom: 4px;
}

.bn-project-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(180px, 1fr));
  gap: 10px;
  margin: 8px 0 14px 0;
}

.bn-project-card {
  min-height: 132px;
  padding: 14px;
  border: 1px solid var(--bn-border);
  background: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 14px rgba(20, 32, 51, 0.05);
}

.bn-project-card-selected {
  border-color: var(--bn-primary);
  box-shadow: 0 0 0 2px rgba(31, 111, 139, 0.12);
}

.bn-project-name {
  color: var(--bn-text);
  font-size: 16px;
  font-weight: 780;
}

.bn-project-slug {
  margin: 2px 0 8px 0;
  color: var(--bn-primary-dark);
  font-size: 12px;
  font-weight: 720;
}

.bn-card-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
}

.bn-control-band {
  padding: 12px;
  border: 1px solid var(--bn-border);
  background: #fbfcfe;
  border-radius: 8px;
}

.bn-inline-hint {
  padding: 9px 11px;
  border: 1px solid var(--bn-border);
  background: #f8fafc;
  border-radius: 8px;
  color: var(--bn-muted);
  font-size: 12px;
  line-height: 1.4;
}

.bn-hint-info {
  border-color: #bfdbfe;
  background: #eff6ff;
  color: #1e3a8a;
}

.bn-hint-warn {
  border-color: #fed7aa;
  background: #fff7ed;
  color: #9a3412;
}

.bn-hint-danger {
  border-color: #f3b6b2;
  background: #fff6f5;
  color: var(--bn-negative);
}

.bn-filter-panel {
  padding: 12px;
  border: 1px solid var(--bn-border);
  background: #fbfcfe;
  border-radius: 8px;
  margin: 8px 0 12px 0;
}

.bn-filter-panel .form {
  gap: 8px !important;
}

.bn-danger-zone {
  border-color: #f3b6b2 !important;
  background: #fffafa !important;
}

.bn-health-panel {
  display: grid;
  grid-template-columns: repeat(2, minmax(220px, 1fr));
  gap: 10px;
  margin: 8px 0 14px 0;
}

.bn-health-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: center;
  padding: 12px 14px;
  border: 1px solid var(--bn-border);
  background: #ffffff;
  border-radius: 8px;
}

.bn-health-label {
  color: var(--bn-muted);
  font-size: 12px;
  font-weight: 720;
}

.bn-health-value {
  margin-top: 2px;
  color: var(--bn-text);
  font-size: 14px;
  font-weight: 720;
}

.bn-validation-grid {
  align-items: stretch;
}

.bn-media-panel,
.bn-sidebar-panel,
.bn-report-panel,
.bn-admin-panel {
  background: var(--bn-panel);
  border: 1px solid var(--bn-border);
  border-radius: 8px;
  box-shadow: var(--bn-shadow);
  padding: 14px;
}

.bn-media-panel .gradio-image,
.bn-media-panel .gradio-audio {
  border-radius: 8px !important;
}

.bn-action-row button {
  min-height: 46px !important;
  border-radius: 7px !important;
  font-weight: 760 !important;
}

.bn-action-row button:nth-child(1) {
  background: var(--bn-positive) !important;
  border-color: var(--bn-positive) !important;
}

.bn-action-row button:nth-child(2) {
  background: #fff1f0 !important;
  border-color: #f3b6b2 !important;
  color: var(--bn-negative) !important;
}

.bn-action-row button:nth-child(3) {
  background: #fff7ed !important;
  border-color: #fed7aa !important;
  color: var(--bn-warning) !important;
}

.bn-action-row button:nth-child(4) {
  background: #eff6ff !important;
  border-color: #bfdbfe !important;
  color: var(--bn-info) !important;
}

.bn-status-strip {
  padding: 10px 12px;
  border-radius: 8px;
  border: 1px solid var(--bn-border);
  background: #f8fafc;
  color: var(--bn-text);
}

.bn-kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(130px, 1fr));
  gap: 10px;
  margin: 6px 0 14px 0;
}

.bn-kpi-card {
  min-height: 82px;
  padding: 12px 14px;
  border-radius: 8px;
  border: 1px solid var(--bn-border);
  background: #ffffff;
}

.bn-kpi-label {
  color: var(--bn-muted);
  font-size: 12px;
  font-weight: 700;
}

.bn-kpi-value {
  margin-top: 4px;
  color: var(--bn-text);
  font-size: 26px;
  line-height: 1.05;
  font-weight: 800;
}

.bn-kpi-hint {
  margin-top: 4px;
  color: var(--bn-muted);
  font-size: 12px;
}

.bn-kpi-positive {
  border-color: #b7dfc9;
  background: #f2fbf6;
}

.bn-kpi-negative {
  border-color: #f3b6b2;
  background: #fff6f5;
}

.bn-kpi-warning {
  border-color: #fed7aa;
  background: #fffaf4;
}

.bn-kpi-info {
  border-color: #bfdbfe;
  background: #f5f9ff;
}

.bn-compact-note {
  color: var(--bn-muted);
  font-size: 12px;
  line-height: 1.4;
}

.bn-dataframe .wrap,
.bn-dataframe table {
  font-size: 13px !important;
}

.bn-dataframe th {
  background: #f1f5f9 !important;
  color: #344054 !important;
  font-weight: 750 !important;
}

textarea,
input,
select {
  border-radius: 7px !important;
}

@media (max-width: 900px) {
  .bn-app-header {
    flex-direction: column;
    align-items: flex-start;
  }

  .bn-header-status {
    justify-content: flex-start;
  }

  .bn-kpi-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .bn-project-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .bn-health-panel {
    grid-template-columns: 1fr;
  }

  .bn-action-row {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  }
}

@media (max-width: 560px) {
  .bn-kpi-grid {
    grid-template-columns: 1fr;
  }

  .bn-project-grid {
    grid-template-columns: 1fr;
  }

  .bn-brand-title {
    font-size: 22px;
  }
}
"""


def app_header_html(state_backend_message: str = "") -> str:
    backend_text = "Supabase ready" if "Supabase state backend enabled" in state_backend_message else "Filesystem state"
    backend_class = "bn-pill-ok" if "Supabase state backend enabled" in state_backend_message else "bn-pill-warn"
    return f"""
    <div class="bn-app-header">
      <div>
        <div class="bn-brand-kicker">BirdNET validation workspace</div>
        <div class="bn-brand-title">BirdNET Validator</div>
        <div class="bn-brand-subtitle">Collaborative review for audio segments, project teams, and validation progress.</div>
      </div>
      <div class="bn-header-status">
        <span class="bn-pill {backend_class}">{backend_text}</span>
        <span class="bn-pill">Hugging Face datasets</span>
        <span class="bn-pill">Multi-validator workflow</span>
      </div>
    </div>
    """
