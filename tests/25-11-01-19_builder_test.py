from pathlib import Path
from unittest.mock import Mock

import pytest

from parallel_developer import cli


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

    monkeypatch.setattr(
        "parallel_developer.cli.TmuxLayoutManager",
        lambda session_name, worker_count, monitor, root_path, **_: tmux,
    )
    monkeypatch.setattr(
        "parallel_developer.cli.WorktreeManager", lambda **_: worktree
    )
    monkeypatch.setattr(
        "parallel_developer.cli.CodexMonitor",
        lambda logs_dir, session_map_path, poll_interval=1.0: monitor,
    )
    monkeypatch.setattr("parallel_developer.cli.LogManager", lambda **_: log_manager)
    monkeypatch.setattr(
        "parallel_developer.cli.Orchestrator",
        lambda **kwargs: orchestrator_instance,
    )

    result = cli.build_orchestrator(worker_count=3, log_dir=None)

    assert result is orchestrator_instance
    tmux.ensure_layout.assert_not_called()
