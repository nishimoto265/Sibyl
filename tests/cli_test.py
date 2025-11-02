from pathlib import Path
from unittest.mock import Mock

import pytest

from parallel_developer.cli import InteractiveCLI, SessionMode
from parallel_developer.orchestrator import CycleArtifact, OrchestrationResult
from parallel_developer.session_manifest import ManifestStore


@pytest.fixture
def manifest_store(tmp_path):
    store_dir = tmp_path / "manifests"
    return ManifestStore(base_dir=store_dir)


def test_parallel_command_updates_worker_count(manifest_store, tmp_path):
    cli = InteractiveCLI(
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )
    assert cli._config.worker_count == 3
    cli._handle_command("/parallel 5")
    assert cli._config.worker_count == 5
    cli._handle_command("/mode main")
    assert cli._config.mode == SessionMode.MAIN


def test_status_command_outputs_information(manifest_store, tmp_path, capsys):
    cli = InteractiveCLI(
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )
    cli._handle_command("/status")
    captured = capsys.readouterr().out
    assert "tmux session" in captured
    assert cli._config.tmux_session in captured


def test_handle_instruction_runs_builder_and_saves_manifest(manifest_store, tmp_path):
    logs_root = tmp_path / "logs"

    artifact = CycleArtifact(
        main_session_id="session-main",
        worker_sessions={},
        boss_session_id=None,
        worker_paths={},
        boss_path=None,
        instruction="",
        tmux_session="parallel-dev-test",
    )
    orchestrator = Mock()
    orchestrator.run_cycle.return_value = OrchestrationResult(
        selected_session="session-main",
        sessions_summary={"main": {"selected": True}},
        artifact=artifact,
    )
    jsonl_path = logs_root / "cycles" / "test.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text('{"type":"instruction","instruction":"Implement feature X"}\n', encoding="utf-8")
    artifact.log_paths["jsonl"] = jsonl_path

    captured_kwargs = {}

    def builder(**kwargs):
        captured_kwargs.update(kwargs)
        return orchestrator

    cli = InteractiveCLI(
        orchestrator_builder=builder,
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )
    cli._config.logs_root = logs_root

    cli._handle_instruction("Implement feature X")

    assert orchestrator.run_cycle.called
    assert captured_kwargs["worker_count"] == cli._config.worker_count
    assert captured_kwargs["session_name"] == cli._config.tmux_session
    sessions = manifest_store.list_sessions()
    assert len(sessions) == 1
    manifest = manifest_store.load_manifest(sessions[0].session_id)
    assert manifest.latest_instruction == "Implement feature X"
    assert manifest.conversation_log.endswith(".jsonl")
    conversation_path = Path(manifest.conversation_log)
    assert conversation_path.exists()
