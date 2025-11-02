import asyncio
import platform
import shlex
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from parallel_developer.cli import CLIController, SessionMode, TmuxAttachManager
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
    controller._attach_manager.session_exists = Mock(return_value=True)
    controller._attach_manager.is_attached = Mock(side_effect=AssertionError("should not be called"))
    controller._attach_manager.attach = mock_attach

    _run_async(controller.handle_input("/attach"))

    mock_attach.assert_called_once_with(controller._config.tmux_session, workdir=tmp_path)
    assert any("tmuxセッション" in payload.get("text", "") for event, payload in events if event == "log")


def test_tmux_attach_manager_macos(monkeypatch, tmp_path):
    manager = TmuxAttachManager()
    recorded = {}

    def fake_run(command, check=False):
        recorded["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(subprocess, "run", fake_run)

    session = "parallel-dev-test"
    manager.attach(session, workdir=tmp_path)

    assert recorded["command"][0] == "osascript"
    assert recorded["command"][1] == "-e"
    script = recorded["command"][2]
    expected_cmd = f"tmux attach -t {shlex.quote(session)}"
    assert expected_cmd in script
    assert "activate" in script


def test_tmux_attach_manager_linux(monkeypatch, tmp_path):
    manager = TmuxAttachManager()
    commands = []

    def fake_run(command, check=False):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(subprocess, "run", fake_run)

    session = "parallel-dev-test"
    manager.attach(session, workdir=tmp_path)

    assert commands
    command = commands[0]
    assert command[:4] == ["gnome-terminal", "--", "bash", "-lc"]
    expected_cmd = f"tmux attach -t {shlex.quote(session)}"
    assert command[4] == expected_cmd


def test_tmux_attach_manager_linux_fallback(monkeypatch, tmp_path):
    manager = TmuxAttachManager()
    commands = []

    def fake_run(command, check=False):
        commands.append(command)
        if command[0] == "gnome-terminal":
            raise FileNotFoundError
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(shutil, "which", lambda _: "/bin/bash")

    session = "parallel-dev-test"
    manager.attach(session, workdir=tmp_path)

    assert commands[-1][0] == "bash"
    expected_cmd = f"tmux attach -t {shlex.quote(session)}"
    assert commands[-1][2] == expected_cmd


def test_attach_auto_mode_skips_when_already_attached(monkeypatch, manifest_store, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )

    controller._attach_manager.is_attached = Mock(return_value=True)
    controller._attach_manager.attach = Mock()
    controller._attach_manager.session_exists = Mock(return_value=True)

    _run_async(controller.handle_input("/attach auto"))
    assert controller._attach_mode == "auto"

    controller._attach_manager.attach.assert_not_called()
    controller._attach_manager.is_attached.assert_not_called()
    assert any("モードを auto に設定しました" in payload.get("text", "") for event, payload in events if event == "log")


def test_attach_manual_mode_ignores_detection(monkeypatch, manifest_store, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )

    _run_async(controller.handle_input("/attach manual"))
    assert controller._attach_mode == "manual"

    controller._attach_manager.is_attached = Mock(return_value=True)
    controller._attach_manager.attach = Mock(return_value=SimpleNamespace(returncode=0))
    controller._attach_manager.session_exists = Mock(return_value=True)

    _run_async(controller._handle_attach_command(force=False))

    controller._attach_manager.attach.assert_called_once()
    assert any("接続しました" in payload.get("text", "") for event, payload in events if event == "log")


def test_attach_mode_persists_between_runs(tmp_path, monkeypatch):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    store = ManifestStore(tmp_path / "manifests")

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=store,
        worktree_root=tmp_path,
    )

    controller._attach_manager.session_exists = Mock(return_value=True)

    _run_async(controller.handle_input("/attach manual"))
    settings_path = tmp_path / ".parallel-dev" / "settings.json"
    assert settings_path.exists()

    controller2 = CLIController(
        event_handler=lambda *_: None,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=store,
        worktree_root=tmp_path,
    )

    assert controller2._attach_mode == "manual"


def test_tmux_attach_manager_is_attached(monkeypatch):
    manager = TmuxAttachManager()

    def fake_run(command, check=False, stdout=None, stderr=None, text=False):
        assert command[:3] == ["tmux", "display-message", "-t"]
        return SimpleNamespace(returncode=0, stdout="1\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert manager.is_attached("parallel-dev-test") is True


def test_tmux_attach_manager_is_not_attached(monkeypatch):
    manager = TmuxAttachManager()

    def fake_run(command, check=False, stdout=None, stderr=None, text=False):
        return SimpleNamespace(returncode=0, stdout="0\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert manager.is_attached("parallel-dev-test") is False


def test_attach_skips_when_session_missing(monkeypatch, manifest_store, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )

    controller._attach_manager.session_exists = Mock(return_value=False)
    controller._attach_manager.attach = Mock()

    async def fast_sleep(_):
        return None

    monkeypatch.setattr("parallel_developer.cli.asyncio.sleep", fast_sleep)

    _run_async(controller._handle_attach_command(force=False))

    controller._attach_manager.attach.assert_not_called()
    assert any("見つかりません" in payload.get("text", "") for event, payload in events if event == "log")


def test_auto_attach_after_instruction(monkeypatch, manifest_store, tmp_path):
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

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **_: orchestrator,
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )
    controller._config.logs_root = logs_root

    controller._attach_manager.session_exists = Mock(return_value=True)
    controller._attach_manager.is_attached = Mock(return_value=False)
    controller._attach_manager.attach = Mock(return_value=SimpleNamespace(returncode=0))

    _run_async(controller.handle_input("/attach auto"))
    assert controller._attach_mode == "auto"
    assert controller._attach_manager.attach.call_count == 0

    _run_async(controller.handle_input("Implement feature Y"))

    assert controller._attach_manager.attach.call_count == 1
    assert any("接続しました" in payload.get("text", "") for event, payload in events if event == "log")
