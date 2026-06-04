CRITICAL_HEAD_HTML = """
<style>
:root {
  --bn-bg: #f5f7fb;
  --bn-shell-width: min(1540px, calc(100vw - 32px));
  --bn-content-width: clamp(1200px, 78vw, 1490px);
  --bn-login-width: min(680px, calc(100vw - 32px));
  color-scheme: light;
}

html,
body,
gradio-app,
main.app,
.wrap,
.contain,
.gradio-container {
  background: var(--bn-bg) !important;
  color-scheme: light !important;
}

.gradio-container {
  box-sizing: border-box !important;
  width: calc(100% - 32px) !important;
  max-width: var(--bn-shell-width) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

.bn-shell {
  box-sizing: border-box !important;
  width: 100% !important;
  max-width: var(--bn-shell-width) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

.bn-app-header,
.bn-tabs,
.bn-panel,
.bn-login-row {
  box-sizing: border-box !important;
  width: 100% !important;
  max-width: var(--bn-content-width) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

.bn-login-row {
  justify-content: center !important;
}

.bn-login-panel {
  box-sizing: border-box !important;
  flex: 0 1 var(--bn-login-width) !important;
  max-width: var(--bn-login-width) !important;
  min-width: 0 !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

.bn-oauth-login-html {
  box-sizing: border-box !important;
  width: 100% !important;
  max-width: var(--bn-login-width) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

.bn-hf-oauth-link {
  box-sizing: border-box !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 10px !important;
  width: 100% !important;
  min-height: 48px !important;
  padding: 12px 18px !important;
  border: 1px solid #d0d7de !important;
  border-radius: 8px !important;
  background: #ffffff !important;
  color: #142033 !important;
  font-weight: 760 !important;
  text-decoration: none !important;
  white-space: nowrap !important;
  box-shadow: 0 1px 2px rgba(20, 32, 51, 0.05) !important;
}

.bn-hf-oauth-mark {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 26px !important;
  height: 26px !important;
  border-radius: 999px !important;
  background: #ffd166 !important;
  color: #503500 !important;
  font-size: 11px !important;
  font-weight: 850 !important;
  letter-spacing: 0 !important;
}

@media (max-width: 1280px) {
  :root {
    --bn-content-width: calc(100vw - 32px);
  }
}
</style>
"""


APP_CSS = """
:root {
  --bn-bg: #f5f7fb;
  --bn-shell-width: min(1540px, calc(100vw - 32px));
  --bn-content-width: clamp(1200px, 78vw, 1490px);
  --bn-login-width: min(680px, calc(100vw - 32px));
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
  color-scheme: light;
}

html,
body,
gradio-app,
main.app,
.wrap,
.contain,
.gradio-container {
  background: var(--bn-bg) !important;
  color: var(--bn-text) !important;
  color-scheme: light !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.gradio-container {
  width: calc(100% - 32px) !important;
  max-width: var(--bn-shell-width) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

.bn-shell {
  width: 100% !important;
  max-width: var(--bn-shell-width) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  padding-top: 8px;
}

.bn-login-row {
  justify-content: center !important;
  width: 100% !important;
  max-width: var(--bn-content-width) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

.bn-login-spacer {
  display: none !important;
}

.bn-login-panel {
  flex: 0 1 var(--bn-login-width) !important;
  max-width: var(--bn-login-width) !important;
  min-width: 0 !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

.bn-login-panel > *,
.bn-login-panel .block,
.bn-login-panel .form {
  width: 100% !important;
}

.bn-app-header {
  box-sizing: border-box;
  display: flex;
  justify-content: space-between;
  gap: 20px;
  align-items: center;
  width: 100% !important;
  max-width: var(--bn-content-width) !important;
  padding: 18px 22px;
  margin: 0 auto 16px auto !important;
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

.bn-tabs {
  box-sizing: border-box;
  width: 100% !important;
  max-width: var(--bn-content-width) !important;
  margin-left: auto !important;
  margin-right: auto !important;
}

.bn-tabs .tabitem {
  width: 100% !important;
}

.bn-tabs button,
.tabs button {
  border-radius: 6px 6px 0 0 !important;
  font-weight: 650 !important;
}

.bn-oauth-login-button button,
button.bn-oauth-login-button {
  width: 100% !important;
  max-width: var(--bn-login-width) !important;
  margin-left: auto !important;
  margin-right: auto !important;
  min-height: 48px !important;
  border: 1px solid #f97316 !important;
  border-radius: 8px !important;
  background: #f97316 !important;
  color: #ffffff !important;
  font-weight: 780 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 8px !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

.bn-oauth-login-button button:hover,
button.bn-oauth-login-button:hover {
  border-color: #ea580c !important;
  background: #ea580c !important;
  color: #ffffff !important;
}

.bn-oauth-login-button button *,
button.bn-oauth-login-button * {
  white-space: nowrap !important;
}

.bn-hf-oauth-link {
  box-sizing: border-box !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 10px !important;
  width: 100% !important;
  max-width: var(--bn-login-width) !important;
  min-height: 48px !important;
  padding: 12px 18px !important;
  border: 1px solid #d0d7de !important;
  border-radius: 8px !important;
  background: #ffffff !important;
  color: var(--bn-text) !important;
  font-weight: 760 !important;
  text-decoration: none !important;
  white-space: nowrap !important;
  box-shadow: 0 1px 2px rgba(20, 32, 51, 0.05) !important;
  transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease !important;
}

.bn-hf-oauth-link:hover,
.bn-hf-oauth-link:focus-visible {
  border-color: #f59e0b !important;
  background: #fffaf0 !important;
  color: var(--bn-text) !important;
  box-shadow: 0 6px 18px rgba(20, 32, 51, 0.08) !important;
  text-decoration: none !important;
}

.bn-hf-oauth-link:active {
  transform: translateY(1px) !important;
}

.bn-hf-oauth-mark {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  flex: 0 0 auto !important;
  width: 26px !important;
  height: 26px !important;
  border-radius: 999px !important;
  background: #ffd166 !important;
  color: #503500 !important;
  font-size: 11px !important;
  font-weight: 850 !important;
  letter-spacing: 0 !important;
  line-height: 1 !important;
}

.bn-hf-oauth-label {
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

.bn-hf-oauth-link-signed-in .bn-hf-oauth-mark {
  background: #e0f2fe !important;
  color: var(--bn-info) !important;
}

.bn-panel {
  box-sizing: border-box;
  width: 100%;
  max-width: var(--bn-content-width);
  margin-left: auto;
  margin-right: auto;
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

.bn-project-context {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-start;
  padding: 14px;
  margin: 8px 0 12px 0;
  border: 1px solid var(--bn-border);
  background: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 14px rgba(20, 32, 51, 0.05);
}

.bn-project-context-title {
  margin-top: 2px;
  color: var(--bn-text);
  font-size: 20px;
  line-height: 1.18;
  font-weight: 800;
}

.bn-project-context-subtitle {
  margin-top: 4px;
  color: var(--bn-muted);
  font-size: 13px;
  word-break: break-word;
}

.bn-invite-callout {
  padding: 12px 14px;
  margin: 8px 0 12px 0;
  border: 1px solid #bfdbfe;
  background: #eff6ff;
  border-radius: 8px;
}

.bn-control-band {
  padding: 12px;
  border: 1px solid var(--bn-border);
  background: #fbfcfe;
  border-radius: 8px;
}

.bn-spacer {
  height: 10px;
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

.bn-clean-slider,
.bn-clean-slider .block,
.bn-clean-slider .form,
.bn-clean-slider .styler,
.bn-clean-slider > div {
  background: #ffffff !important;
}

.bn-clean-slider input,
.bn-clean-slider [data-testid="number-input"] {
  background: #ffffff !important;
}

.bn-sidebar-panel .block,
.bn-sidebar-panel .form,
.bn-sidebar-panel .styler,
.bn-sidebar-panel .wrap,
.bn-sidebar-panel .contain,
.bn-sidebar-panel .gradio-slider,
.bn-sidebar-panel .gradio-slider > div,
.bn-sidebar-panel .gradio-slider label,
.bn-sidebar-panel .gradio-slider input,
.bn-sidebar-panel .gradio-slider [data-testid="number-input"] {
  background: #ffffff !important;
}

.bn-danger-zone {
  border-color: #f3b6b2 !important;
  background: #fffafa !important;
}

.bn-selected-card {
  padding: 14px;
  border: 1px solid var(--bn-border);
  background: linear-gradient(180deg, #ffffff 0%, #f8fbfc 100%);
  border-radius: 8px;
  margin: 0 0 12px 0;
}

.bn-selected-topline {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: flex-start;
}

.bn-selected-species {
  margin-top: 2px;
  color: var(--bn-text);
  font-size: 21px;
  line-height: 1.15;
  font-weight: 820;
}

.bn-selected-meta {
  display: grid;
  grid-template-columns: repeat(4, minmax(90px, 1fr));
  gap: 8px;
  margin-top: 12px;
}

.bn-selected-meta span {
  padding: 8px 10px;
  border: 1px solid var(--bn-border);
  background: #ffffff;
  border-radius: 7px;
  color: var(--bn-text);
  font-size: 13px;
  font-weight: 720;
}

.bn-selected-meta strong {
  display: block;
  color: var(--bn-muted);
  font-size: 11px;
  font-weight: 720;
  margin-bottom: 2px;
}

.bn-selected-audio {
  margin-top: 10px;
  color: var(--bn-muted);
  font-size: 12px;
  word-break: break-word;
}

.bn-queue-preview {
  margin-top: 12px;
  padding: 12px;
  border: 1px solid var(--bn-border);
  background: #fbfcfe;
  border-radius: 8px;
}

.bn-queue-preview-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 8px;
  color: var(--bn-muted);
  font-size: 12px;
  font-weight: 760;
  text-transform: uppercase;
}

.bn-queue-card {
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr) auto;
  gap: 10px;
  align-items: center;
  padding: 9px 10px;
  border: 1px solid transparent;
  border-radius: 7px;
}

.bn-queue-card + .bn-queue-card {
  margin-top: 4px;
}

.bn-queue-card-selected {
  border-color: var(--bn-primary);
  background: #eef8fb;
}

.bn-queue-card-index {
  width: 28px;
  height: 28px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: #ffffff;
  border: 1px solid var(--bn-border);
  color: var(--bn-primary-dark);
  font-size: 12px;
  font-weight: 800;
}

.bn-queue-card-main {
  min-width: 0;
}

.bn-queue-card-species {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--bn-text);
  font-size: 13px;
  font-weight: 760;
}

.bn-queue-card-audio {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: var(--bn-muted);
  font-size: 11px;
}

.bn-queue-card-meta {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.bn-queue-card-meta span {
  padding: 4px 7px;
  border: 1px solid var(--bn-border);
  background: #ffffff;
  border-radius: 999px;
  color: var(--bn-muted);
  font-size: 11px;
  font-weight: 720;
}

.bn-queue-footnote,
.bn-empty-mini {
  margin-top: 8px;
  color: var(--bn-muted);
  font-size: 12px;
}

.bn-coverage-panel {
  padding: 12px;
  margin: 8px 0 14px 0;
  border: 1px solid var(--bn-border);
  background: #ffffff;
  border-radius: 8px;
  box-shadow: 0 4px 14px rgba(20, 32, 51, 0.04);
}

.bn-coverage-row {
  padding: 9px 0;
}

.bn-coverage-row + .bn-coverage-row {
  border-top: 1px solid #eef2f6;
}

.bn-coverage-row-head {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  color: var(--bn-text);
  font-size: 13px;
  font-weight: 760;
}

.bn-coverage-row-head span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bn-coverage-row-head strong {
  color: var(--bn-primary-dark);
  font-size: 12px;
}

.bn-coverage-track {
  height: 8px;
  margin-top: 7px;
  overflow: hidden;
  background: #edf2f7;
  border-radius: 999px;
}

.bn-coverage-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--bn-primary) 0%, var(--bn-positive) 100%);
  border-radius: 999px;
}

.bn-coverage-meta {
  margin-top: 5px;
  color: var(--bn-muted);
  font-size: 11px;
}

.bn-activity-table-wrap {
  margin-top: 10px;
  overflow: hidden;
  border: 1px solid var(--bn-border);
  border-radius: 8px;
  background: #ffffff;
}

.bn-activity-table {
  width: 100%;
  min-width: 520px;
  border-collapse: separate;
  border-spacing: 0;
  border: 0 !important;
  background: #ffffff;
  color: var(--bn-text);
  font-size: 13px;
}

.bn-activity-table th,
.bn-activity-table td {
  padding: 9px 11px;
  border: 0 !important;
  border-bottom: 1px solid var(--bn-border) !important;
  text-align: left;
  vertical-align: top;
}

.bn-activity-table th {
  color: #344054;
  background: #f1f5f9;
  font-weight: 750;
}

.bn-activity-table th + th,
.bn-activity-table td + td {
  border-left: 1px solid var(--bn-border) !important;
}

.bn-activity-table thead th:first-child {
  border-top-left-radius: 8px;
}

.bn-activity-table thead th:last-child {
  border-top-right-radius: 8px;
}

.bn-activity-table tbody tr:last-child td {
  border-bottom: 0 !important;
}

.bn-activity-empty {
  color: var(--bn-muted);
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

.bn-admin-panel .block,
.bn-admin-panel .form,
.bn-card-body {
  background: #ffffff !important;
}

.bn-admin-panel > .styler,
.bn-admin-panel .styler {
  background: var(--bn-bg) !important;
}

.bn-admin-panel {
  box-shadow: none !important;
}

.bn-admin-panel .block:has(button),
.bn-admin-panel .form:has(button),
.bn-report-panel .block:has(button),
.bn-report-panel .form:has(button),
.bn-clean-button-row,
.bn-clean-button-row .block,
.bn-clean-button-row .form,
.bn-clean-action,
.bn-soft-action {
  background: transparent !important;
}

.bn-card-body {
  border: 1px solid var(--bn-border) !important;
  border-radius: 8px !important;
  padding: 12px !important;
}

.bn-admin-section {
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  background: var(--bn-bg) !important;
  gap: 8px !important;
}

.bn-admin-section,
.bn-admin-section .block:has(.bn-admin-action),
.bn-admin-section .form:has(.bn-admin-action),
.bn-admin-action-row,
.bn-admin-action-row > *,
.bn-admin-action-row .block,
.bn-admin-action-row .form {
  background: var(--bn-bg) !important;
}

div.row.bn-admin-action-row {
  background-color: var(--bn-bg) !important;
  background-image: linear-gradient(var(--bn-bg), var(--bn-bg)) !important;
}

div.row.bn-admin-action-row > :not(button) {
  background-color: var(--bn-bg) !important;
}

.bn-admin-access-section,
.bn-admin-pending-section {
  width: 100% !important;
}

.column:has(> .bn-admin-access-section):has(> .bn-admin-pending-section) {
  display: grid !important;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  align-items: start;
  gap: 14px !important;
}

.column:has(> .bn-admin-access-section):has(> .bn-admin-pending-section) > :not(.bn-admin-access-section):not(.bn-admin-pending-section) {
  grid-column: 1 / -1;
}

.bn-soft-action button,
button.bn-soft-action {
  min-height: 42px !important;
  border: 1px solid #93c5fd !important;
  border-radius: 8px !important;
  background: #dbeafe !important;
  color: #075985 !important;
  font-weight: 760 !important;
}

.bn-soft-action button:hover,
button.bn-soft-action:hover {
  border-color: #60a5fa !important;
  background: #bfdbfe !important;
}

.bn-clean-action button.secondary,
button.bn-clean-action.secondary {
  border: 1px solid #93c5fd !important;
  border-radius: 8px !important;
  background: #dbeafe !important;
  color: #075985 !important;
  font-weight: 760 !important;
}

.bn-clean-action button.secondary:hover,
button.bn-clean-action.secondary:hover {
  border-color: #60a5fa !important;
  background: #bfdbfe !important;
}

.bn-orange-action button,
button.bn-orange-action {
  min-height: 42px !important;
  border: 1px solid #f97316 !important;
  border-radius: 8px !important;
  background: #f97316 !important;
  color: #ffffff !important;
  font-weight: 760 !important;
}

.bn-orange-action button:hover,
button.bn-orange-action:hover {
  border-color: #ea580c !important;
  background: #ea580c !important;
  color: #ffffff !important;
}

.bn-admin-action button,
button.bn-admin-action {
  min-height: 48px !important;
  border-radius: 8px !important;
  font-weight: 790 !important;
}

.bn-admin-action-orange button,
button.bn-admin-action-orange {
  border-color: #f97316 !important;
  background: #f97316 !important;
  color: #ffffff !important;
}

.bn-admin-action-orange button:hover,
button.bn-admin-action-orange:hover {
  border-color: #ea580c !important;
  background: #ea580c !important;
}

.bn-admin-action-blue button,
button.bn-admin-action-blue {
  border-color: #93c5fd !important;
  background: #dbeafe !important;
  color: #075985 !important;
}

.bn-admin-action-blue button:hover,
button.bn-admin-action-blue:hover {
  border-color: #60a5fa !important;
  background: #bfdbfe !important;
}

.bn-admin-action-red button,
button.bn-admin-action-red {
  border-color: #dc2626 !important;
  background: #dc2626 !important;
  color: #ffffff !important;
}

.bn-admin-action-red button:hover,
button.bn-admin-action-red:hover {
  border-color: #b91c1c !important;
  background: #b91c1c !important;
}

.bn-delete-project-action button {
  min-height: 54px !important;
}

.bn-report-download-section {
  margin-top: 12px;
  background: var(--bn-bg) !important;
  gap: 8px !important;
}

.bn-report-download-panel,
.bn-report-download-panel .block,
.bn-report-download-panel .form {
  background: #ffffff !important;
}

.bn-report-download-panel {
  box-shadow: none !important;
}

.bn-report-download-panel > .styler,
.bn-report-download-panel .styler {
  background: var(--bn-bg) !important;
}

.bn-report-download-section,
.bn-report-download-action-row,
.bn-report-download-action-row > *,
.bn-report-download-action-row .block,
.bn-report-download-action-row .form {
  background: var(--bn-bg) !important;
}

div.row.bn-report-download-action-row {
  background-color: var(--bn-bg) !important;
  background-image: linear-gradient(var(--bn-bg), var(--bn-bg)) !important;
}

.bn-report-download-action button,
button.bn-report-download-action {
  min-height: 48px !important;
  border-radius: 8px !important;
  font-weight: 790 !important;
}

.bn-report-download-action-orange button,
button.bn-report-download-action-orange {
  border-color: #f97316 !important;
  background: #f97316 !important;
  color: #ffffff !important;
}

.bn-report-download-action-orange button:hover,
button.bn-report-download-action-orange:hover {
  border-color: #ea580c !important;
  background: #ea580c !important;
}

.bn-report-download-action-blue button,
button.bn-report-download-action-blue {
  border-color: #93c5fd !important;
  background: #dbeafe !important;
  color: #075985 !important;
}

.bn-report-download-action-blue button:hover,
button.bn-report-download-action-blue:hover {
  border-color: #60a5fa !important;
  background: #bfdbfe !important;
}

.bn-autodownload-target {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  overflow: hidden !important;
  clip-path: inset(50%) !important;
}

.bn-media-panel .gradio-image,
.bn-media-panel .gradio-audio {
  border-radius: 8px !important;
}

.bn-media-panel [data-testid="image"],
.bn-media-panel [data-testid="audio"] {
  max-width: 100%;
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

.bn-dataframe {
  overflow-x: auto !important;
}

.bn-dataframe table {
  min-width: 920px !important;
}

.bn-dataframe th {
  background: #f1f5f9 !important;
  color: #344054 !important;
  font-weight: 750 !important;
}

.bn-polished-dataframe {
  overflow: hidden !important;
  border: 1px solid var(--bn-border) !important;
  border-radius: 8px !important;
  background: #eef4fb !important;
  box-shadow: 0 10px 24px rgba(16, 24, 40, 0.05) !important;
  padding: 0 !important;
}

.bn-polished-dataframe .table-container {
  margin: 0 !important;
  padding: 0 !important;
  background: #ffffff !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}

.bn-polished-dataframe .header-row {
  position: static !important;
  display: flex !important;
  align-items: center !important;
  min-height: 40px !important;
  margin: 0 !important;
  padding: 10px 14px !important;
  border: 0 !important;
  border-bottom: 1px solid #dbe5ef !important;
  background: #eef4fb !important;
  color: #40566f !important;
  font-size: 13px !important;
  font-weight: 650 !important;
  line-height: 1.2 !important;
}

.bn-polished-dataframe .header-row .label,
.bn-polished-dataframe .header-row .label p {
  margin: 0 !important;
  padding: 0 !important;
  color: inherit !important;
  font: inherit !important;
  line-height: inherit !important;
}

.bn-polished-dataframe .wrap,
.bn-polished-dataframe .table-wrap,
.bn-polished-dataframe [data-testid="dataframe"] {
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 0 8px 8px !important;
  background: #ffffff !important;
}

.bn-polished-dataframe,
.bn-polished-dataframe .wrap,
.bn-polished-dataframe .table-wrap,
.bn-polished-dataframe [data-testid="dataframe"],
.bn-polished-dataframe [role="grid"] {
  max-width: 100% !important;
  overflow-x: hidden !important;
}

.bn-polished-dataframe table {
  width: 100% !important;
  min-width: 0 !important;
  table-layout: fixed !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
}

.bn-polished-dataframe th,
.bn-polished-dataframe td {
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
  padding: 9px 8px !important;
  border-color: #e6edf5 !important;
  vertical-align: middle !important;
}

.bn-polished-dataframe th {
  background: #eef4fb !important;
  color: #1f344d !important;
  font-size: 12px !important;
  letter-spacing: 0 !important;
  text-transform: uppercase !important;
}

.bn-polished-dataframe td {
  color: #102033 !important;
  font-size: 13px !important;
}

.bn-polished-dataframe tbody tr:nth-child(even) td {
  background: #fbfdff !important;
}

.bn-polished-dataframe tbody tr:hover td {
  background: #eff6ff !important;
}

.bn-polished-dataframe * {
  scrollbar-width: none !important;
}

.bn-polished-dataframe *::-webkit-scrollbar {
  display: none !important;
}

#bn-admin-projects-table th:nth-child(1),
#bn-admin-projects-table td:nth-child(1) {
  width: 13% !important;
}

#bn-admin-projects-table th:nth-child(2),
#bn-admin-projects-table td:nth-child(2) {
  width: 17% !important;
}

#bn-admin-projects-table th:nth-child(3),
#bn-admin-projects-table td:nth-child(3) {
  width: 28% !important;
}

#bn-admin-projects-table th:nth-child(4),
#bn-admin-projects-table td:nth-child(4),
#bn-admin-projects-table th:nth-child(5),
#bn-admin-projects-table td:nth-child(5) {
  width: 13% !important;
}

#bn-admin-projects-table th:nth-child(6),
#bn-admin-projects-table td:nth-child(6),
#bn-admin-projects-table th:nth-child(7),
#bn-admin-projects-table td:nth-child(7) {
  width: 8% !important;
}

#bn-admin-pending-invites-table th:nth-child(1),
#bn-admin-pending-invites-table td:nth-child(1),
#bn-admin-pending-invites-table th:nth-child(2),
#bn-admin-pending-invites-table td:nth-child(2),
#bn-admin-pending-invites-table th:nth-child(4),
#bn-admin-pending-invites-table td:nth-child(4) {
  width: 18% !important;
}

#bn-admin-pending-invites-table th:nth-child(3),
#bn-admin-pending-invites-table td:nth-child(3),
#bn-admin-pending-invites-table th:nth-child(6),
#bn-admin-pending-invites-table td:nth-child(6) {
  width: 12% !important;
}

#bn-admin-pending-invites-table th:nth-child(5),
#bn-admin-pending-invites-table td:nth-child(5) {
  width: 22% !important;
}

#bn-admin-projects-table td:nth-child(1),
#bn-admin-projects-table td:nth-child(3),
#bn-admin-pending-invites-table td:nth-child(1),
#bn-admin-pending-invites-table td:nth-child(2),
#bn-admin-pending-invites-table td:nth-child(5) {
  color: #475467 !important;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
  font-size: 12px !important;
}

#bn-admin-projects-table td:nth-child(6),
#bn-admin-projects-table td:nth-child(7),
#bn-admin-pending-invites-table td:nth-child(3),
#bn-admin-pending-invites-table td:nth-child(6) {
  color: #0b5e86 !important;
  font-weight: 750 !important;
}

.bn-validation-dataframe {
  overflow: hidden !important;
  border: 1px solid var(--bn-border) !important;
  border-radius: 8px !important;
  background: #eef4fb !important;
  box-shadow: 0 10px 24px rgba(16, 24, 40, 0.05) !important;
}

#bn-validation-queue-table {
  padding: 0 !important;
}

#bn-validation-queue-table .table-container {
  margin: 0 !important;
  padding: 0 !important;
  background: #ffffff !important;
  border-radius: 8px !important;
  overflow: hidden !important;
}

#bn-validation-queue-table .header-row {
  position: static !important;
  display: flex !important;
  align-items: center !important;
  min-height: 40px !important;
  margin: 0 !important;
  padding: 10px 14px !important;
  border: 0 !important;
  border-bottom: 1px solid #dbe5ef !important;
  background: #eef4fb !important;
  color: #40566f !important;
  font-size: 13px !important;
  font-weight: 650 !important;
  line-height: 1.2 !important;
}

#bn-validation-queue-table .header-row .label,
#bn-validation-queue-table .header-row .label p {
  margin: 0 !important;
  padding: 0 !important;
  color: inherit !important;
  font: inherit !important;
  line-height: inherit !important;
}

#bn-validation-queue-table .wrap,
#bn-validation-queue-table .table-wrap,
#bn-validation-queue-table [data-testid="dataframe"] {
  margin: 0 !important;
  padding: 0 !important;
  border: 0 !important;
  border-radius: 0 0 8px 8px !important;
  background: #ffffff !important;
}

#bn-validation-queue-table,
#bn-validation-queue-table .wrap,
#bn-validation-queue-table .table-wrap,
#bn-validation-queue-table [data-testid="dataframe"],
#bn-validation-queue-table [role="grid"] {
  max-width: 100% !important;
  overflow-x: hidden !important;
}

#bn-validation-queue-table table {
  width: 100% !important;
  min-width: 0 !important;
  table-layout: fixed !important;
  border-collapse: separate !important;
  border-spacing: 0 !important;
}

#bn-validation-queue-table th,
#bn-validation-queue-table td {
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
  padding: 9px 8px !important;
  border-color: #e6edf5 !important;
  vertical-align: middle !important;
}

#bn-validation-queue-table th:nth-child(1),
#bn-validation-queue-table td:nth-child(1) {
  width: 8% !important;
}

#bn-validation-queue-table th:nth-child(2),
#bn-validation-queue-table td:nth-child(2) {
  width: 24% !important;
}

#bn-validation-queue-table th:nth-child(3),
#bn-validation-queue-table td:nth-child(3) {
  width: 21% !important;
}

#bn-validation-queue-table th:nth-child(4),
#bn-validation-queue-table td:nth-child(4) {
  width: 10% !important;
}

#bn-validation-queue-table th:nth-child(5),
#bn-validation-queue-table td:nth-child(5),
#bn-validation-queue-table th:nth-child(6),
#bn-validation-queue-table td:nth-child(6) {
  width: 8% !important;
}

#bn-validation-queue-table th:nth-child(7),
#bn-validation-queue-table td:nth-child(7) {
  width: 14% !important;
}

#bn-validation-queue-table th:nth-child(8),
#bn-validation-queue-table td:nth-child(8) {
  width: 7% !important;
}

#bn-validation-queue-table th:nth-child(9),
#bn-validation-queue-table td:nth-child(9),
#bn-validation-queue-table th:nth-child(10),
#bn-validation-queue-table td:nth-child(10) {
  display: none !important;
}

#bn-validation-queue-table th {
  background: #eef4fb !important;
  color: #1f344d !important;
  font-size: 12px !important;
  letter-spacing: 0 !important;
  text-transform: uppercase !important;
}

#bn-validation-queue-table td {
  color: #102033 !important;
  font-size: 13px !important;
}

#bn-validation-queue-table tbody tr:nth-child(even) td {
  background: #fbfdff !important;
}

#bn-validation-queue-table tbody tr:hover td {
  background: #eff6ff !important;
  cursor: pointer !important;
}

#bn-validation-queue-table td:nth-child(1),
#bn-validation-queue-table td:nth-child(2) {
  color: #475467 !important;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
  font-size: 12px !important;
}

#bn-validation-queue-table td:nth-child(4),
#bn-validation-queue-table td:nth-child(5),
#bn-validation-queue-table td:nth-child(6),
#bn-validation-queue-table td:nth-child(8) {
  text-align: right !important;
  font-variant-numeric: tabular-nums !important;
}

#bn-validation-queue-table td:nth-child(7) {
  color: #0b5e86 !important;
  font-weight: 750 !important;
}

#bn-validation-queue-table * {
  scrollbar-width: none !important;
}

#bn-validation-queue-table *::-webkit-scrollbar {
  display: none !important;
}

#bn-species-status-payload {
  display: none !important;
}

#bn-species-filter li[data-testid="dropdown-option"] {
  box-sizing: border-box !important;
  margin: 4px 8px !important;
  padding: 8px 10px !important;
  border: 1px solid var(--bn-species-status-border, var(--bn-border)) !important;
  border-left-width: 1px !important;
  border-radius: 8px !important;
  background: var(--bn-species-status-bg, #ffffff) !important;
  box-shadow: inset 6px 0 0 var(--bn-species-status-accent, #94a3b8) !important;
  color: var(--bn-species-status-text, var(--bn-text)) !important;
  font-weight: 650 !important;
}

#bn-species-filter li[data-testid="dropdown-option"] .inner-item {
  display: none !important;
}

#bn-species-filter li[data-testid="dropdown-option"].active,
#bn-species-filter li[data-testid="dropdown-option"].selected,
#bn-species-filter li[data-testid="dropdown-option"]:hover {
  background: var(--bn-species-status-hover, #eff6ff) !important;
  color: var(--bn-species-status-text, var(--bn-text)) !important;
}

#bn-species-filter li[data-testid="dropdown-option"].bn-species-option-unvalidated,
#bn-species-filter li[data-testid="dropdown-option"][data-species-status="unvalidated"] {
  --bn-species-status-border: #cbd5e1;
  --bn-species-status-accent: #94a3b8;
  --bn-species-status-bg: #f8fafc;
  --bn-species-status-hover: #eef2f7;
  --bn-species-status-text: #243348;
  border-color: #cbd5e1 !important;
}

#bn-species-filter li[data-testid="dropdown-option"].bn-species-option-partial,
#bn-species-filter li[data-testid="dropdown-option"][data-species-status="partial"] {
  --bn-species-status-border: #f59e0b;
  --bn-species-status-accent: #f59e0b;
  --bn-species-status-bg: #fff7ed;
  --bn-species-status-hover: #ffedd5;
  --bn-species-status-text: #7c2d12;
  border-color: #f59e0b !important;
}

#bn-species-filter li[data-testid="dropdown-option"].bn-species-option-complete,
#bn-species-filter li[data-testid="dropdown-option"][data-species-status="complete"] {
  --bn-species-status-border: #16a34a;
  --bn-species-status-accent: #16a34a;
  --bn-species-status-bg: #ecfdf3;
  --bn-species-status-hover: #dcfce7;
  --bn-species-status-text: #14532d;
  border-color: #16a34a !important;
}

#bn-species-filter.bn-species-selected-unvalidated .wrap-inner {
  border-color: #cbd5e1 !important;
  box-shadow: 0 0 0 1px rgba(148, 163, 184, 0.3) !important;
}

#bn-species-filter.bn-species-selected-partial .wrap-inner {
  border-color: #f59e0b !important;
  box-shadow: 0 0 0 1px rgba(245, 158, 11, 0.35) !important;
}

#bn-species-filter.bn-species-selected-complete .wrap-inner {
  border-color: #16a34a !important;
  box-shadow: 0 0 0 1px rgba(22, 163, 74, 0.35) !important;
}

#bn-corrected-species-error-panel {
  margin-top: -6px !important;
  margin-bottom: 8px !important;
}

.bn-corrected-species-alert {
  border: 1px solid #ef4444;
  border-left: 6px solid #dc2626;
  border-radius: 8px;
  background: #fff5f5;
  color: #7f1d1d;
  padding: 12px 14px;
  box-shadow: 0 10px 24px rgba(127, 29, 29, 0.08);
}

.bn-corrected-species-alert strong {
  display: block;
  color: #991b1b;
  font-weight: 800;
  margin-bottom: 3px;
}

.bn-corrected-species-alert span {
  color: #7f1d1d;
  font-size: 0.94rem;
}

textarea,
input,
select {
  border-radius: 7px !important;
}

textarea,
input,
select,
button {
  color-scheme: light !important;
}

@media (max-width: 1280px) {
  :root {
    --bn-content-width: calc(100vw - 32px);
  }
}

@media (max-width: 900px) {
  :root,
  body,
  gradio-app,
  main.app,
  .wrap,
  .contain,
  .gradio-container,
  .gradio-container .dark,
  .gradio-container [data-theme],
  .gradio-container .app,
  .gradio-container .main,
  .gradio-container .block,
  .gradio-container .form,
  .gradio-container .styler,
  .gradio-container .wrap,
  .gradio-container .contain,
  .gradio-container .container,
  .gradio-container .input-container,
  .gradio-container .wrap-inner,
  .gradio-container .secondary-wrap {
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #f5f7fb !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #d7e2ed !important;
    --block-info-text-color: #667085 !important;
    --body-background-fill: #f5f7fb !important;
    --body-text-color: #142033 !important;
    --body-text-color-subdued: #667085 !important;
    --button-primary-background-fill: #f97316 !important;
    --button-primary-background-fill-hover: #ea580c !important;
    --button-primary-text-color: #ffffff !important;
    --button-secondary-background-fill: #dbeafe !important;
    --button-secondary-background-fill-hover: #bfdbfe !important;
    --button-secondary-border-color: #93c5fd !important;
    --button-secondary-text-color: #075985 !important;
    --checkbox-background-color-selected: #f97316 !important;
    --input-background-fill: #ffffff !important;
    --input-background-fill-focus: #ffffff !important;
    --input-border-color: #d7e2ed !important;
    --input-border-color-focus: #93c5fd !important;
    --input-placeholder-color: #98a2b3 !important;
    --input-shadow: none !important;
    --input-shadow-focus: 0 0 0 3px rgba(59, 130, 246, 0.12) !important;
    --link-text-color: #1f6f8b !important;
    --neutral-50: #f8fafc !important;
    --neutral-100: #eef2f6 !important;
    --neutral-200: #d7e2ed !important;
    --neutral-300: #c4cfdd !important;
    --neutral-400: #98a2b3 !important;
    --neutral-500: #667085 !important;
    --neutral-600: #475467 !important;
    --neutral-700: #344054 !important;
    --neutral-800: #243348 !important;
    --neutral-900: #142033 !important;
    --neutral-950: #101828 !important;
    --panel-background-fill: #ffffff !important;
    --table-border-color: #d7e2ed !important;
    --table-even-background-fill: #ffffff !important;
    --table-odd-background-fill: #fbfdff !important;
    --table-row-focus: #eff6ff !important;
  }

  .gradio-container,
  .gradio-container *,
  .gradio-container *::before,
  .gradio-container *::after {
    color-scheme: light !important;
  }

  .bn-app-header {
    flex-direction: column;
    align-items: flex-start;
  }

  html,
  body,
  gradio-app,
  main.app,
  .wrap,
  .contain,
  .gradio-container,
  .bn-shell {
    background: var(--bn-bg) !important;
    color: var(--bn-text) !important;
    color-scheme: light !important;
  }

  .bn-tabs .tab-nav,
  .tabs .tab-nav {
    background: transparent !important;
  }

  .tabs [role="menu"],
  .tabs [role="menu"] *,
  .tabs .overflow-menu,
  .tabs .overflow-menu *,
  .tabs .overflow-menu button,
  .tabs .overflow-menu a,
  .gradio-container .options,
  .gradio-container .options *,
  .gradio-container .options li,
  .gradio-container .options ul,
  .gradio-container [role="listbox"],
  .gradio-container [role="listbox"] *,
  .gradio-container [role="option"],
  .gradio-container [role="menu"],
  .gradio-container [role="menu"] *,
  .gradio-container .menu,
  .gradio-container .menu * {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-tabs button,
  .tabs button,
  .bn-tabs [role="tab"],
  .tabs [role="tab"] {
    background: transparent !important;
    color: var(--bn-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-tabs button.selected,
  .tabs button.selected,
  .bn-tabs [role="tab"][aria-selected="true"],
  .tabs [role="tab"][aria-selected="true"] {
    color: #f97316 !important;
    -webkit-text-fill-color: #f97316 !important;
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

  .bn-project-context {
    flex-direction: column;
  }

  .bn-health-panel {
    grid-template-columns: 1fr;
  }

  .bn-selected-meta {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .bn-action-row {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  }

  .bn-validation-grid {
    gap: 12px !important;
  }

  .bn-media-panel,
  .bn-sidebar-panel,
  .bn-report-panel,
  .bn-admin-panel,
  .bn-panel {
    padding: 12px;
  }

  .bn-admin-access-section,
  .bn-admin-pending-section {
    display: block !important;
    width: 100% !important;
    margin-left: 0 !important;
  }

  .column:has(> .bn-admin-access-section):has(> .bn-admin-pending-section) {
    grid-template-columns: 1fr;
  }

  .bn-brand-kicker,
  .bn-section-title,
  .bn-brand-title,
  .bn-empty-title,
  .bn-selected-species,
  .bn-project-context-title,
  .bn-project-name,
  .bn-kpi-value,
  .bn-health-value {
    color: var(--bn-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-brand-subtitle,
  .bn-compact-note,
  .bn-project-context-subtitle,
  .bn-selected-audio,
  .bn-kpi-hint,
  .bn-kpi-label,
  .bn-health-label {
    color: var(--bn-muted) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-muted) !important;
  }

  .bn-queue-preview-head,
  .bn-queue-preview-head span,
  .bn-coverage-row-head,
  .bn-coverage-row-head span,
  .bn-coverage-row-head strong,
  .bn-activity-table,
  .bn-activity-table th,
  .bn-activity-table td,
  .bn-health-row,
  .bn-health-row * {
    color: var(--bn-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-coverage-row-head strong {
    color: var(--bn-primary-dark) !important;
    -webkit-text-fill-color: var(--bn-primary-dark) !important;
  }

  .bn-coverage-meta,
  .bn-queue-footnote,
  .bn-empty-mini,
  .bn-activity-empty {
    color: var(--bn-muted) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-muted) !important;
  }

  .bn-panel [data-testid="markdown"],
  .bn-panel [data-testid="markdown"] *,
  .bn-login-panel [data-testid="markdown"],
  .bn-login-panel [data-testid="markdown"] *,
  .bn-media-panel [data-testid="markdown"],
  .bn-media-panel [data-testid="markdown"] *,
  .bn-sidebar-panel [data-testid="markdown"],
  .bn-sidebar-panel [data-testid="markdown"] *,
  .bn-admin-panel [data-testid="markdown"],
  .bn-admin-panel [data-testid="markdown"] *,
  .bn-report-panel [data-testid="markdown"],
  .bn-report-panel [data-testid="markdown"] * {
    color: var(--bn-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-panel,
  .bn-panel-soft,
  .bn-login-panel,
  .bn-media-panel,
  .bn-sidebar-panel,
  .bn-report-panel,
  .bn-admin-panel,
  .bn-selected-card,
  .bn-project-context,
  .bn-project-card,
  .bn-kpi-card,
  .bn-health-row,
  .bn-empty-state,
  .bn-status-strip {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    color-scheme: light !important;
  }

  .gradio-container :is(
    .gradio-textbox,
    .gradio-dropdown,
    .gradio-slider,
    .gradio-checkbox,
    .gradio-radio,
    .gradio-date,
    .gradio-datetime,
    .gradio-number,
    .gradio-dataframe,
    .gradio-image,
    .gradio-audio
  ),
  .gradio-container :is(
    .block,
    .form,
    .styler,
    .wrap,
    .contain,
    .container,
    .input-container,
    .wrap-inner,
    .secondary-wrap
  ):has(input),
  .gradio-container :is(
    .block,
    .form,
    .styler,
    .wrap,
    .contain,
    .container,
    .input-container,
    .wrap-inner,
    .secondary-wrap
  ):has(textarea),
  .gradio-container :is(
    .block,
    .form,
    .styler,
    .wrap,
    .contain,
    .container,
    .input-container,
    .wrap-inner,
    .secondary-wrap
  ):has(select),
  .gradio-container :is(
    .block,
    .form,
    .styler,
    .wrap,
    .contain,
    .container,
    .input-container,
    .wrap-inner,
    .secondary-wrap
  ):has([data-testid="dropdown"]),
  .gradio-container :is(
    .block,
    .form,
    .styler,
    .wrap,
    .contain,
    .container,
    .input-container,
    .wrap-inner,
    .secondary-wrap
  ):has([data-testid="textbox"]),
  .gradio-container :is(
    .block,
    .form,
    .styler,
    .wrap,
    .contain,
    .container,
    .input-container,
    .wrap-inner,
    .secondary-wrap
  ):has([data-testid="number-input"]) {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    color-scheme: light !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .gradio-container .block.padded,
  .gradio-container .block.padded.auto-margin,
  .gradio-container .block.padded.svelte-11xb1hd,
  .gradio-container .block.padded.auto-margin.svelte-11xb1hd,
  .gradio-container .block.padded > .container,
  .gradio-container .block.padded .container,
  .gradio-container .block.padded .wrap,
  .gradio-container .block.padded .wrap-inner,
  .gradio-container .block.padded .secondary-wrap,
  .gradio-container .block.padded .input-container {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    border-color: var(--bn-border) !important;
    color-scheme: light !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .gradio-container input,
  .gradio-container textarea,
  .gradio-container select,
  .gradio-container [contenteditable="true"],
  .gradio-container [data-testid="textbox"],
  .gradio-container [data-testid="number-input"],
  .gradio-container [data-testid="dropdown"] {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    border-color: var(--bn-border) !important;
    color-scheme: light !important;
    caret-color: var(--bn-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .gradio-container textarea.scroll-hide[data-testid="textbox"],
  .gradio-container textarea.scroll-hide.svelte-173056l[data-testid="textbox"],
  .gradio-container input.scroll-hide[data-testid="password"],
  .gradio-container input.scroll-hide.svelte-173056l[data-testid="password"],
  .gradio-container input.scroll-hide[data-testid="textbox"],
  .gradio-container input.scroll-hide.svelte-173056l[data-testid="textbox"],
  .gradio-container input.scroll-hide[data-testid="number-input"],
  .gradio-container input.scroll-hide.svelte-173056l[data-testid="number-input"],
  .gradio-container input.scroll-hide[type="text"],
  .gradio-container input.scroll-hide[type="password"] {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    border-color: var(--bn-border) !important;
    color-scheme: light !important;
    caret-color: var(--bn-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .gradio-container textarea.scroll-hide[data-testid="textbox"]::placeholder,
  .gradio-container input.scroll-hide[data-testid="password"]::placeholder,
  .gradio-container input.scroll-hide[data-testid="textbox"]::placeholder,
  .gradio-container input.scroll-hide[data-testid="number-input"]::placeholder,
  .gradio-container input.scroll-hide[type="text"]::placeholder,
  .gradio-container input.scroll-hide[type="password"]::placeholder {
    color: #98a2b3 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #98a2b3 !important;
  }

  .gradio-container input::placeholder,
  .gradio-container textarea::placeholder {
    color: #98a2b3 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #98a2b3 !important;
  }

  .gradio-container label,
  .gradio-container .label,
  .gradio-container [data-testid="block-info"],
  .gradio-container [data-testid="block-info"] * {
    color: var(--bn-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .gradio-container label[data-testid$="-radio-label"],
  .gradio-container label[data-testid$="-radio-label"] span,
  .gradio-container label.svelte-k79vs1[data-testid$="-radio-label"],
  .gradio-container label.svelte-k79vs1[data-testid$="-radio-label"] span {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    border-color: var(--bn-border) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .gradio-container label[data-testid$="-radio-label"].selected,
  .gradio-container label[data-testid$="-radio-label"].selected span,
  .gradio-container label.svelte-k79vs1[data-testid$="-radio-label"].selected,
  .gradio-container label.svelte-k79vs1[data-testid$="-radio-label"].selected span {
    background: #fff7ed !important;
    color: #9a3412 !important;
    border-color: #fed7aa !important;
    -webkit-text-fill-color: #9a3412 !important;
  }

  .bn-card-pills,
  .bn-card-pills *,
  .bn-pill,
  .bn-pill * {
    opacity: 1 !important;
  }

  .bn-pill,
  .bn-pill strong {
    -webkit-text-fill-color: currentColor !important;
  }

  .bn-dataframe,
  .bn-dataframe *,
  .bn-polished-dataframe,
  .bn-polished-dataframe *,
  .bn-admin-dataframe,
  .bn-admin-dataframe * {
    color-scheme: light !important;
  }

  .bn-dataframe table,
  .bn-dataframe tbody,
  .bn-dataframe tr,
  .bn-dataframe td,
  .bn-dataframe svelte-virtual-table-viewport,
  .bn-polished-dataframe table,
  .bn-polished-dataframe tbody,
  .bn-polished-dataframe tr,
  .bn-polished-dataframe td,
  .bn-polished-dataframe svelte-virtual-table-viewport {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-dataframe thead,
  .bn-dataframe th,
  .bn-polished-dataframe thead,
  .bn-polished-dataframe th {
    background: #f1f5f9 !important;
    color: #1f344d !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #1f344d !important;
  }

  .bn-activity-table,
  .bn-activity-table tbody,
  .bn-activity-table tr,
  .bn-activity-table td {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-activity-table th {
    background: #f1f5f9 !important;
    color: #1f344d !important;
    -webkit-text-fill-color: #1f344d !important;
  }

  .bn-login-panel,
  .bn-login-panel .block,
  .bn-login-panel .form,
  .bn-login-panel .wrap,
  .bn-login-panel .contain,
  .bn-login-panel .styler,
  .bn-login-panel .input-container,
  .bn-login-panel .wrap-inner,
  .bn-login-panel .secondary-wrap,
  .bn-login-panel label,
  .bn-login-panel textarea,
  .bn-login-panel input,
  .bn-login-panel select,
  .bn-login-panel [contenteditable="true"],
  .bn-login-panel [data-testid="textbox"],
  .bn-login-panel [data-testid="number-input"],
  .bn-login-panel [data-testid="dropdown"] {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    color-scheme: light !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-login-panel h1,
  .bn-login-panel h2,
  .bn-login-panel h3,
  .bn-login-panel p,
  .bn-login-panel label,
  .bn-login-panel span,
  .bn-login-panel .label,
  .bn-login-panel .prose,
  .bn-login-panel .markdown {
    color: var(--bn-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-login-panel textarea::placeholder,
  .bn-login-panel input::placeholder {
    color: #98a2b3 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #98a2b3 !important;
  }

  .bn-admin-panel .block,
  .bn-admin-panel .form,
  .bn-admin-panel .styler,
  .bn-admin-panel .wrap,
  .bn-admin-panel .contain,
  .bn-admin-panel .input-container,
  .bn-admin-panel .wrap-inner,
  .bn-admin-panel .secondary-wrap,
  .bn-admin-panel label,
  .bn-admin-panel textarea,
  .bn-admin-panel input,
  .bn-admin-panel select,
  .bn-admin-panel [contenteditable="true"],
  .bn-admin-panel [data-testid="textbox"],
  .bn-admin-panel [data-testid="dropdown"],
  .bn-admin-panel [data-testid="number-input"],
  .bn-report-panel .block,
  .bn-report-panel .form,
  .bn-report-panel .styler,
  .bn-report-panel .wrap,
  .bn-report-panel .contain,
  .bn-report-panel .input-container,
  .bn-report-panel .wrap-inner,
  .bn-report-panel .secondary-wrap,
  .bn-report-panel label,
  .bn-report-panel textarea,
  .bn-report-panel input,
  .bn-report-panel select,
  .bn-report-panel [contenteditable="true"],
  .bn-report-panel [data-testid="textbox"],
  .bn-report-panel [data-testid="dropdown"],
  .bn-report-panel [data-testid="number-input"] {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    color-scheme: light !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-admin-panel textarea::placeholder,
  .bn-admin-panel input::placeholder,
  .bn-report-panel textarea::placeholder,
  .bn-report-panel input::placeholder {
    color: #98a2b3 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #98a2b3 !important;
  }

  .bn-sidebar-panel .block,
  .bn-sidebar-panel .form,
  .bn-sidebar-panel .styler,
  .bn-sidebar-panel .wrap,
  .bn-sidebar-panel .contain,
  .bn-sidebar-panel .input-container,
  .bn-sidebar-panel .wrap-inner,
  .bn-sidebar-panel .secondary-wrap,
  .bn-sidebar-panel label,
  .bn-sidebar-panel textarea,
  .bn-sidebar-panel input,
  .bn-sidebar-panel select,
  .bn-sidebar-panel [contenteditable="true"],
  .bn-sidebar-panel [data-testid="textbox"],
  .bn-sidebar-panel [data-testid="dropdown"],
  .bn-sidebar-panel [data-testid="number-input"],
  .bn-media-panel .gradio-dropdown,
  .bn-media-panel .gradio-dropdown .wrap,
  .bn-media-panel .gradio-dropdown .wrap-inner,
  .bn-media-panel .gradio-dropdown .input-container,
  .bn-media-panel .gradio-dropdown input,
  .bn-media-panel .gradio-dropdown [data-testid="dropdown"],
  .bn-media-panel .gradio-dropdown [data-testid="textbox"] {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    color-scheme: light !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-media-panel .block,
  .bn-media-panel .form,
  .bn-media-panel .styler,
  .bn-media-panel .wrap,
  .bn-media-panel .contain,
  .bn-media-panel .input-container,
  .bn-media-panel .wrap-inner,
  .bn-media-panel .secondary-wrap,
  .bn-media-panel .label-wrap,
  .bn-media-panel .label-wrap *,
  .bn-media-panel [data-testid="block-info"],
  .bn-media-panel [data-testid="block-info"] *,
  .bn-media-panel label,
  .bn-media-panel textarea,
  .bn-media-panel input,
  .bn-media-panel select,
  .bn-media-panel [contenteditable="true"] {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    color-scheme: light !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-sidebar-panel textarea::placeholder,
  .bn-sidebar-panel input::placeholder,
  .bn-media-panel .gradio-dropdown input::placeholder {
    color: #98a2b3 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #98a2b3 !important;
  }

  .bn-sidebar-panel .options,
  .bn-sidebar-panel .options ul,
  .bn-sidebar-panel [role="listbox"],
  .bn-sidebar-panel li[data-testid="dropdown-option"],
  .bn-media-panel .gradio-dropdown .options,
  .bn-media-panel .gradio-dropdown .options ul,
  .bn-media-panel .gradio-dropdown [role="listbox"],
  .bn-media-panel .gradio-dropdown li[data-testid="dropdown-option"] {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    color-scheme: light !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-media-panel .gradio-image,
  .bn-media-panel .gradio-audio,
  .bn-media-panel .gradio-checkbox,
  .bn-media-panel .gradio-image > *,
  .bn-media-panel .gradio-audio > *,
  .bn-media-panel .gradio-checkbox > *,
  .bn-media-panel .gradio-image .wrap,
  .bn-media-panel .gradio-audio .wrap,
  .bn-media-panel .gradio-checkbox .wrap,
  .bn-media-panel .gradio-image .container,
  .bn-media-panel .gradio-audio .container,
  .bn-media-panel .gradio-image .image-container,
  .bn-media-panel .gradio-audio .audio-container,
  .bn-media-panel .gradio-image .empty,
  .bn-media-panel .gradio-audio .empty,
  .bn-media-panel .gradio-image .icon-wrap,
  .bn-media-panel .gradio-audio .icon-wrap,
  .bn-media-panel .gradio-image .icon,
  .bn-media-panel .gradio-audio .icon,
  .bn-media-panel .gradio-image [data-testid="image"],
  .bn-media-panel .gradio-audio [data-testid="audio"],
  .bn-media-panel .gradio-image [data-testid="image"] *,
  .bn-media-panel .gradio-audio [data-testid="audio"] * {
    background: #ffffff !important;
    color: var(--bn-text) !important;
    color-scheme: light !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
    border-color: var(--bn-border) !important;
  }

  .bn-media-panel .gradio-image iframe,
  .bn-media-panel .gradio-audio iframe {
    background: #ffffff !important;
    color-scheme: light !important;
  }

  .bn-media-panel .empty,
  .bn-media-panel .unpadded_box,
  .bn-media-panel .padded_box,
  .bn-media-panel .audio-container,
  .bn-media-panel .image-container,
  .bn-media-panel .file-preview,
  .bn-media-panel .download,
  .bn-media-panel .toolbar {
    background: #ffffff !important;
    border-color: var(--bn-border) !important;
    box-shadow: none !important;
  }

  .bn-media-panel .gradio-image svg,
  .bn-media-panel .gradio-audio svg {
    color: #667085 !important;
    fill: currentColor !important;
    stroke: currentColor !important;
    opacity: 1 !important;
  }

  .bn-media-panel .gradio-checkbox label,
  .bn-media-panel .gradio-checkbox span,
  .bn-media-panel .gradio-checkbox p {
    color: var(--bn-text) !important;
    opacity: 1 !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  .bn-media-panel .gradio-checkbox input[type="checkbox"] {
    accent-color: #f97316 !important;
    background: #ffffff !important;
  }

  .bn-media-panel .gradio-image button,
  .bn-media-panel .gradio-audio button,
  .bn-media-panel .gradio-image [role="button"],
  .bn-media-panel .gradio-audio [role="button"],
  .bn-media-panel .toolbar button,
  .bn-media-panel .download button {
    background: #ffffff !important;
    color: #344054 !important;
    border-color: var(--bn-border) !important;
    -webkit-text-fill-color: #344054 !important;
  }

  .bn-admin-action,
  .bn-admin-action button,
  button.bn-admin-action,
  .bn-report-download-action,
  .bn-report-download-action button,
  button.bn-report-download-action,
  .bn-clean-action,
  .bn-clean-action button,
  button.bn-clean-action,
  .bn-soft-action,
  .bn-soft-action button,
  button.bn-soft-action {
    min-height: 44px !important;
    border-radius: 8px !important;
    font-weight: 790 !important;
    opacity: 1 !important;
    text-decoration: none !important;
  }

  .bn-admin-action-orange,
  .bn-admin-action-orange button,
  button.bn-admin-action-orange,
  .bn-orange-action,
  .bn-orange-action button,
  button.bn-orange-action,
  .bn-report-download-action-orange,
  .bn-report-download-action-orange button,
  button.bn-report-download-action-orange {
    border-color: #f97316 !important;
    background: #f97316 !important;
    background-image: none !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
  }

  .bn-admin-action-blue,
  .bn-admin-action-blue button,
  button.bn-admin-action-blue,
  .bn-clean-action,
  .bn-clean-action button,
  button.bn-clean-action,
  .bn-soft-action,
  .bn-soft-action button,
  button.bn-soft-action,
  .bn-report-download-action-blue,
  .bn-report-download-action-blue button,
  button.bn-report-download-action-blue {
    border-color: #93c5fd !important;
    background: #dbeafe !important;
    background-image: none !important;
    color: #075985 !important;
    -webkit-text-fill-color: #075985 !important;
  }

  .bn-admin-action-red,
  .bn-admin-action-red button,
  button.bn-admin-action-red {
    border-color: #dc2626 !important;
    background: #dc2626 !important;
    background-image: none !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
  }

  .bn-action-row button,
  .bn-action-row button * {
    opacity: 1 !important;
  }

  .gradio-container button.lg.primary:not(.bn-admin-action):not(.bn-report-download-action):not(.bn-clean-action):not(.bn-orange-action),
  .gradio-container button.lg.primary:not(.bn-admin-action):not(.bn-report-download-action):not(.bn-clean-action):not(.bn-orange-action) *,
  .gradio-container button.lg.secondary:not(.bn-admin-action):not(.bn-report-download-action):not(.bn-clean-action):not(.bn-soft-action),
  .gradio-container button.lg.secondary:not(.bn-admin-action):not(.bn-report-download-action):not(.bn-clean-action):not(.bn-soft-action) * {
    opacity: 1 !important;
    text-decoration: none !important;
  }

  .gradio-container button.lg.primary:not(.bn-admin-action):not(.bn-report-download-action):not(.bn-clean-action):not(.bn-orange-action) {
    min-height: 44px !important;
    border: 1px solid #f97316 !important;
    border-radius: 8px !important;
    background: #f97316 !important;
    background-image: none !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 790 !important;
  }

  .gradio-container button.lg.primary:not(.bn-admin-action):not(.bn-report-download-action):not(.bn-clean-action):not(.bn-orange-action) * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
  }

  .gradio-container button.lg.secondary:not(.bn-admin-action):not(.bn-report-download-action):not(.bn-clean-action):not(.bn-soft-action) {
    min-height: 44px !important;
    border: 1px solid #93c5fd !important;
    border-radius: 8px !important;
    background: #dbeafe !important;
    background-image: none !important;
    color: #075985 !important;
    -webkit-text-fill-color: #075985 !important;
    font-weight: 790 !important;
  }

  .gradio-container button.lg.secondary:not(.bn-admin-action):not(.bn-report-download-action):not(.bn-clean-action):not(.bn-soft-action) * {
    color: #075985 !important;
    -webkit-text-fill-color: #075985 !important;
  }

  #bn-validate-confirm-btn,
  #bn-validate-confirm-btn button,
  button#bn-validate-confirm-btn {
    background: var(--bn-positive) !important;
    border-color: var(--bn-positive) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
  }

  #bn-validate-reject-btn,
  #bn-validate-reject-btn button,
  button#bn-validate-reject-btn {
    background: #fff1f0 !important;
    border-color: #f3b6b2 !important;
    color: var(--bn-negative) !important;
    -webkit-text-fill-color: var(--bn-negative) !important;
  }

  #bn-validate-uncertain-btn,
  #bn-validate-uncertain-btn button,
  button#bn-validate-uncertain-btn {
    background: #fff7ed !important;
    border-color: #fed7aa !important;
    color: var(--bn-warning) !important;
    -webkit-text-fill-color: var(--bn-warning) !important;
  }

  #bn-validate-skip-btn,
  #bn-validate-skip-btn button,
  button#bn-validate-skip-btn {
    background: #eff6ff !important;
    border-color: #bfdbfe !important;
    color: var(--bn-info) !important;
    -webkit-text-fill-color: var(--bn-info) !important;
  }

  #bn-validate-favorite-btn,
  #bn-validate-favorite-btn button,
  button#bn-validate-favorite-btn,
  .bn-action-row button:nth-child(5) {
    background: #f8fafc !important;
    background-color: #f8fafc !important;
    border-color: var(--bn-border-strong) !important;
    color: var(--bn-text) !important;
    -webkit-text-fill-color: var(--bn-text) !important;
  }

  #bn-species-filter .options,
  #bn-species-filter .options ul,
  #bn-species-filter [role="listbox"] {
    background: #ffffff !important;
    color: var(--bn-text) !important;
  }

  #bn-species-filter li[data-testid="dropdown-option"].bn-species-option-unvalidated,
  #bn-species-filter li[data-testid="dropdown-option"][data-species-status="unvalidated"] {
    background: #f8fafc !important;
    color: #243348 !important;
    -webkit-text-fill-color: #243348 !important;
  }

  #bn-species-filter li[data-testid="dropdown-option"].bn-species-option-partial,
  #bn-species-filter li[data-testid="dropdown-option"][data-species-status="partial"] {
    background: #fff7ed !important;
    color: #7c2d12 !important;
    -webkit-text-fill-color: #7c2d12 !important;
  }

  #bn-species-filter li[data-testid="dropdown-option"].bn-species-option-complete,
  #bn-species-filter li[data-testid="dropdown-option"][data-species-status="complete"] {
    background: #ecfdf3 !important;
    color: #14532d !important;
    -webkit-text-fill-color: #14532d !important;
  }

  .bn-sidebar-panel button.lg.primary {
    border: 1px solid #f97316 !important;
    background: #f97316 !important;
    background-color: #f97316 !important;
    background-image: none !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 790 !important;
  }

  .bn-sidebar-panel button.lg.secondary {
    border: 1px solid #93c5fd !important;
    background: #dbeafe !important;
    background-color: #dbeafe !important;
    background-image: none !important;
    color: #075985 !important;
    -webkit-text-fill-color: #075985 !important;
    font-weight: 790 !important;
  }

  .bn-sidebar-panel button.lg.primary *,
  .bn-sidebar-panel button.lg.secondary * {
    color: currentColor !important;
    -webkit-text-fill-color: currentColor !important;
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

  .bn-app-header {
    padding: 14px;
  }

  .bn-selected-card {
    padding: 12px;
  }

  .bn-selected-topline {
    flex-direction: column;
    gap: 8px;
  }

  .bn-selected-species {
    font-size: 18px;
  }

  .bn-selected-meta {
    grid-template-columns: 1fr;
  }

  .bn-action-row {
    grid-template-columns: 1fr !important;
  }

  .bn-action-row button {
    width: 100% !important;
    min-height: 48px !important;
  }

  .bn-queue-card {
    grid-template-columns: 28px minmax(0, 1fr);
  }

  .bn-queue-card-meta {
    grid-column: 2;
    justify-content: flex-start;
  }

  .bn-dataframe table {
    min-width: 780px !important;
  }

  #bn-validation-queue-table table {
    min-width: 0 !important;
  }

  #bn-validation-queue-table th,
  #bn-validation-queue-table td {
    padding: 8px 6px !important;
    font-size: 12px !important;
  }
}
"""


def app_header_html(state_backend_message: str = "") -> str:
    if "HF admin-owned private storage enabled" in state_backend_message:
        backend_text = "Private HF storage ready"
        backend_class = "bn-pill-ok"
    elif "Supabase state backend enabled" in state_backend_message:
        backend_text = "Supabase ready"
        backend_class = "bn-pill-ok"
    else:
        backend_text = "Filesystem state"
        backend_class = "bn-pill-warn"
    return f"""
    <div class="bn-app-header" style="box-sizing:border-box;width:100%;max-width:var(--bn-content-width,1490px);margin:0 auto 16px auto;">
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
