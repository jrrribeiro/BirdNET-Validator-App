CRITICAL_HEAD_HTML = """
<style>
:root {
  --bn-bg: #f5f7fb;
  --bn-shell-width: min(1540px, calc(100vw - 32px));
  --bn-content-width: clamp(1200px, 78vw, 1490px);
  --bn-login-width: min(680px, calc(100vw - 32px));
}

html,
body,
gradio-app,
main.app,
.wrap,
.contain,
.gradio-container {
  background: var(--bn-bg) !important;
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

textarea,
input,
select {
  border-radius: 7px !important;
}

@media (max-width: 1280px) {
  :root {
    --bn-content-width: calc(100vw - 32px);
  }
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
