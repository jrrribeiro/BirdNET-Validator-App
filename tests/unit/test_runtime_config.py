from src.config.runtime_config import RuntimeConfig


def test_runtime_config_reads_hf_project_state_writes_flag(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("BIRDNET_HF_PROJECT_STATE_WRITES_ENABLED", "true")

    config = RuntimeConfig.from_env()

    assert config.hf_project_state_writes_enabled is True


def test_runtime_config_disables_hf_project_state_writes_by_default(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.delenv("BIRDNET_HF_PROJECT_STATE_WRITES_ENABLED", raising=False)

    config = RuntimeConfig.from_env()

    assert config.hf_project_state_writes_enabled is False
