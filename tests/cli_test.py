import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from parallel_developer.cli import CLIController, SessionMode
from parallel_developer.orchestrator import CycleArtifact, OrchestrationResult
from parallel_developer.session_manifest import ManifestStore
from types import SimpleNamespace


@pytest.fixture
def manifest_store(tmp_path):
    store_dir = tmp_path / "manifests"
    return ManifestStore(base_dir=store_dir)


def _run_async(coro):
    return asyncio.run(coro)


def test_parallel_command_updates_worker_count(manifest_store, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )

    assert controller._config.worker_count == 3
    _run_async(controller.handle_input("/parallel 5"))
    assert controller._config.worker_count == 5
    _run_async(controller.handle_input("/mode main"))
    assert controller._config.mode == SessionMode.MAIN
    status_events = [payload for event, payload in events if event == "status"]
    assert status_events


def test_status_command_outputs_information(manifest_store, tmp_path):
    captured = []

    def handler(event_type, payload):
        captured.append((event_type, payload))

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )
    _run_async(controller.handle_input("/status"))
    status_events = [payload for et, payload in captured if et == "status"]
    assert status_events
    assert status_events[-1]["message"] == "待機中"


def test_handle_instruction_runs_builder_and_saves_manifest(manifest_store, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

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
    artifact.selected_session_id = "session-main"
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

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=builder,
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )
    controller._config.logs_root = logs_root

    _run_async(controller.handle_input("Implement feature X"))

    assert orchestrator.run_cycle.called
    assert captured_kwargs["worker_count"] == controller._config.worker_count
    assert captured_kwargs["session_name"] == controller._config.tmux_session
    sessions = manifest_store.list_sessions()
    assert len(sessions) == 1
    manifest = manifest_store.load_manifest(sessions[0].session_id)
    assert manifest.latest_instruction == "Implement feature X"
    assert manifest.conversation_log.endswith(".jsonl")
    conversation_path = Path(manifest.conversation_log)
    assert conversation_path.exists()
    assert controller._last_selected_session == "session-main"


def test_attach_command_invokes_tmux(monkeypatch, manifest_store, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )

    mock_attach = Mock(return_value=SimpleNamespace(returncode=0))
    controller._attach_manager = Mock(attach=mock_attach)

    _run_async(controller.handle_input("/attach"))

    mock_attach.assert_called_once()
    assert any("tmuxセッション" in payload.get("text", "") for event, payload in events if event == "log")
