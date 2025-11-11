from pathlib import Path
from unittest.mock import Mock

import pytest

from parallel_developer import controller


@pytest.fixture(autouse=True)
def isolate_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield


def test_build_orchestrator_wires_dependencies(monkeypatch):
    tmux = Mock(name="TmuxManager")
    worktree = Mock(name="WorktreeManager")
    monitor = Mock(name="CodexMonitor")
    log_manager = Mock(name="LogManager")
    orchestrator_instance = Mock(name="OrchestratorInstance")

    def fake_tmux(**kwargs):
        assert kwargs["session_namespace"] == "namespace"
        assert kwargs["root_path"] == Path.cwd()
        return tmux

    monkeypatch.setattr(
        "parallel_developer.controller.TmuxLayoutManager",
        fake_tmux,
    )

    def fake_worktree(**kwargs):
        assert kwargs["session_namespace"] == "namespace"
        assert kwargs["root"] == Path.cwd()
        assert kwargs["storage_root"] == Path.cwd()
        return worktree

    monkeypatch.setattr(
        "parallel_developer.controller.WorktreeManager", fake_worktree
    )

    def fake_monitor(**kwargs):
        return monitor

    monkeypatch.setattr(
        "parallel_developer.controller.CodexMonitor",
        fake_monitor,
    )
    monkeypatch.setattr("parallel_developer.controller.LogManager", lambda **_: log_manager)
    def fake_orchestrator(**kwargs):
        assert kwargs["boss_mode"].value == "score"
        assert kwargs["merge_mode"].value == "manual"
        return orchestrator_instance

    monkeypatch.setattr(
        "parallel_developer.controller.Orchestrator",
        fake_orchestrator,
    )

    result = controller.build_orchestrator(worker_count=3, log_dir=None, session_namespace="namespace")

    assert result is orchestrator_instance
    tmux.ensure_layout.assert_not_called()
