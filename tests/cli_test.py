import asyncio
import json
from concurrent.futures import Future
import platform
import shlex
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from parallel_developer.controller import CLIController, SessionMode, TmuxAttachManager
from parallel_developer.orchestrator import BossMode, CycleArtifact, OrchestrationResult
from parallel_developer.session_manifest import ManifestStore, PaneRecord, SessionManifest
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


def test_handle_instruction_runs_builder_and_saves_manifest(manifest_store, tmp_path, monkeypatch):
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

    home_dir = tmp_path / "home"
    home_dir.mkdir()
    codex_config = home_dir / ".codex"
    codex_config.mkdir(parents=True, exist_ok=True)
    (codex_config / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home_dir)

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
    codex_home = captured_kwargs["codex_home"]
    assert codex_home.exists()
    assert codex_home != home_dir
    assert (codex_home / ".codex" / "config.json").exists()
    assert (codex_home / ".codex" / ".bootstrap_complete").exists()
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


def test_controller_broadcast_escape(monkeypatch, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(event_handler=handler, worktree_root=tmp_path)

    recorded: list[list[str]] = []

    def fake_run(command, check=False, stdout=None, stderr=None, text=None):
        recorded.append(command)
        if "list-panes" in command:
            return SimpleNamespace(returncode=0, stdout="%0\n%1\n", stderr="")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    controller.broadcast_escape()

    expected_prefix = ["tmux", "list-panes", "-t", controller._config.tmux_session, "-F", "#{pane_id}"]
    assert recorded[0] == expected_prefix
    assert ["tmux", "send-keys", "-t", "%0", "Escape"] in recorded
    assert ["tmux", "send-keys", "-t", "%1", "Escape"] in recorded
    logs = [payload["text"] for event, payload in events if event == "log"]
    assert f"tmuxセッション {controller._config.tmux_session} の 2 個のペインへEscapeを送信しました。" in logs


def test_handle_escape_enters_pause(monkeypatch, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(event_handler=handler, worktree_root=tmp_path)
    controller.broadcast_escape = lambda: None  # type: ignore[assignment]

    controller.handle_escape()

    assert controller._paused is True
    status_messages = [payload["message"] for event, payload in events if event == "status"]
    assert "一時停止モード" in status_messages[-1]
    log_messages = [payload["text"] for event, payload in events if event == "log"]
    assert any("一時停止モード" in msg for msg in log_messages)


def test_handle_escape_reverts_cycle(monkeypatch, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(event_handler=handler, worktree_root=tmp_path)
    controller.broadcast_escape = lambda: None  # type: ignore[assignment]
    controller._cycle_history = [
        {"selected_session": "session-A", "scoreboard": {"main": {}}, "instruction": "first"},
        {"selected_session": "session-B", "scoreboard": {"main": {}}, "instruction": "second"},
    ]
    controller._last_selected_session = "session-B"
    controller._last_scoreboard = {"main": {}}
    controller._paused = True
    controller._running = True
    tmux_called = {}
    class DummyTmux:
        def promote_to_main(self, session_id, pane_id):
            tmux_called["session"] = session_id
            tmux_called["pane"] = pane_id
    controller._last_tmux_manager = DummyTmux()
    controller._tmux_list_panes = lambda: ["%0", "%1", "%2"]


    controller.handle_escape()

    assert controller._paused is False
    assert controller._last_selected_session == "session-A"
    status_messages = [payload["message"] for event, payload in events if event == "status"]
    assert status_messages and status_messages[-1] == "待機中"
    log_messages = [payload["text"] for event, payload in events if event == "log"]

    assert tmux_called.get("session") == "session-A"
    assert tmux_called.get("pane") == "%0"


def test_handle_escape_reverts_initial_cycle(monkeypatch, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(event_handler=handler, worktree_root=tmp_path)
    controller.broadcast_escape = lambda: None  # type: ignore[assignment]
    controller._paused = True
    controller._running = True
    controller._cycle_history = []
    controller._active_main_session_id = "session-main"
    controller._last_selected_session = None

    tmux_called = {}

    class DummyTmux:
        def promote_to_main(self, session_id, pane_id):
            tmux_called["session"] = session_id
            tmux_called["pane"] = pane_id

        def launch_main_session(self, pane_id):
            tmux_called["session"] = None
            tmux_called["pane"] = pane_id

    controller._last_tmux_manager = DummyTmux()
    controller._tmux_list_panes = lambda: ["%0", "%1"]

    controller.handle_escape()

    assert controller._paused is False
    assert controller._last_selected_session == "session-main"
    assert controller._active_main_session_id == "session-main"
    assert tmux_called.get("session") == "session-main"
    assert tmux_called.get("pane") == "%0"
    status_messages = [payload["message"] for event, payload in events if event == "status"]
    assert status_messages and status_messages[-1] == "待機中"


def test_on_main_session_started_sets_reuse_flag(tmp_path):
    controller = CLIController(event_handler=lambda *_: None, worktree_root=tmp_path)
    assert controller._config.reuse_existing_session is False

    controller._on_main_session_started("session-main")

    assert controller._active_main_session_id == "session-main"
    assert controller._config.reuse_existing_session is True


def test_history_navigation(tmp_path):
    controller = CLIController(event_handler=lambda *_: None, worktree_root=tmp_path)

    controller._record_history("first")
    controller._record_history("second")
    controller._record_history("second")  # duplicate ignored
    controller._record_history("third")

    assert controller.history_previous() == "third"
    assert controller.history_previous() == "second"
    assert controller.history_previous() == "first"
    assert controller.history_previous() == "first"  # stays at oldest

    assert controller.history_next() == "second"
    assert controller.history_next() == "third"
    assert controller.history_next() == ""
    assert controller.history_next() == ""

    controller.history_reset()
    assert controller.history_next() == ""


def test_paused_instruction_broadcast(monkeypatch, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(event_handler=handler, worktree_root=tmp_path)
    controller._paused = True

    recorded: list[list[str]] = []

    def fake_run(command, check=False, stdout=None, stderr=None, text=None):
        recorded.append(command)
        if "list-panes" in command:
            return SimpleNamespace(returncode=0, stdout="%0\n%1\n%2\n%3\n", stderr="")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    _run_async(controller.handle_input("echo pause"))

    expected_prefix = ["tmux", "list-panes", "-t", controller._config.tmux_session, "-F", "#{pane_id}"]
    assert expected_prefix in recorded
    assert ["tmux", "send-keys", "-t", "%2", "echo pause", "Enter"] in recorded
    assert ["tmux", "send-keys", "-t", "%3", "echo pause", "Enter"] in recorded
    assert ["tmux", "send-keys", "-t", "%0", "echo pause", "Enter"] not in recorded
    assert ["tmux", "send-keys", "-t", "%1", "echo pause", "Enter"] not in recorded
    log_messages = [payload["text"] for event, payload in events if event == "log"]
    assert any("[pause] 2 ワーカーペイン" in msg for msg in log_messages)
    assert controller._paused is False


def test_done_command_forces_completion(monkeypatch, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(event_handler=handler, worktree_root=tmp_path)
    forced = {}

    class StubOrchestrator:
        def force_complete_workers(self_inner):
            forced["called"] = True
            return 2

    controller._active_orchestrator = StubOrchestrator()

    _run_async(controller.execute_command("/done"))

    assert forced.get("called") is True
    log_messages = [payload["text"] for event, payload in events if event == "log"]
    assert any(" /done " in msg or "/done" in msg for msg in log_messages)


def test_continue_command_sets_future(tmp_path):
    controller = CLIController(event_handler=lambda *_: None, worktree_root=tmp_path)
    future = Future()
    controller._continue_future = future

    _run_async(controller.execute_command("/continue"))

    assert future.done() and future.result() is True


def test_done_command_resolves_continue_future(tmp_path):
    controller = CLIController(event_handler=lambda *_: None, worktree_root=tmp_path)
    future = Future()
    controller._continue_future = future

    _run_async(controller.execute_command("/done"))

    assert future.done() and future.result() is False


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


def test_codex_home_shared_mode(monkeypatch, manifest_store, tmp_path):
    home_dir = tmp_path / "global_home"
    home_dir.mkdir()
    (home_dir / ".codex").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(Path, "home", lambda: home_dir)
    monkeypatch.setenv("PARALLEL_DEV_CODEX_HOME_MODE", "shared")

    captured_kwargs = {}

    def builder(**kwargs):
        captured_kwargs.update(kwargs)
        orchestrator = Mock()
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
        orchestrator.run_cycle.return_value = OrchestrationResult(
            selected_session="session-main",
            sessions_summary={"main": {"selected": True}},
            artifact=artifact,
        )
        return orchestrator

    controller = CLIController(
        event_handler=lambda *_: None,
        orchestrator_builder=builder,
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )
    controller._config.logs_root = tmp_path / "logs"
    _run_async(controller.handle_input("Implement feature Y"))

    assert captured_kwargs["codex_home"] == home_dir
    assert not (home_dir / ".codex" / ".bootstrap_complete").exists()


def test_codex_home_env_override(monkeypatch, manifest_store, tmp_path):
    base_override = tmp_path / "override" / "{session}"
    monkeypatch.setenv("PARALLEL_DEV_CODEX_HOME", str(base_override))

    home_dir = tmp_path / "base_home"
    home_dir.mkdir()
    (home_dir / ".codex").mkdir(parents=True, exist_ok=True)
    (home_dir / ".codex" / "profile.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: home_dir)

    captured_kwargs = {}

    def builder(**kwargs):
        captured_kwargs.update(kwargs)
        orchestrator = Mock()
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
        orchestrator.run_cycle.return_value = OrchestrationResult(
            selected_session="session-main",
            sessions_summary={"main": {"selected": True}},
            artifact=artifact,
        )
        return orchestrator

    controller = CLIController(
        event_handler=lambda *_: None,
        orchestrator_builder=builder,
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )
    controller._config.logs_root = tmp_path / "logs"
    session_specific = Path(str(base_override).replace("{session}", controller._session_namespace))

    _run_async(controller.handle_input("Implement feature Z"))

    assert captured_kwargs["codex_home"] == session_specific
    assert (session_specific / ".codex" / "profile.json").exists()


def test_codex_home_command_updates_mode(tmp_path):
    controller = CLIController(event_handler=lambda *_: None, worktree_root=tmp_path)
    assert controller._codex_home_mode == "session"
    _run_async(controller.handle_input("/codexhome shared"))
    assert controller._codex_home_mode == "shared"
    settings_path = tmp_path / ".parallel-dev" / "settings.json"
    assert settings_path.exists()
    data = json.loads(settings_path.read_text(encoding="utf-8"))
    assert data["codex_home_mode"] == "shared"


def test_codex_home_command_blocked_by_env(tmp_path, monkeypatch):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    monkeypatch.setenv("PARALLEL_DEV_CODEX_HOME_MODE", "shared")
    controller = CLIController(event_handler=handler, worktree_root=tmp_path)

    _run_async(controller.handle_input("/codexhome session"))
    # 設定は変わらず session のまま
    assert controller._codex_home_mode == "session"
    assert any("環境変数" in payload.get("text", "") for event, payload in events if event == "log")


def test_boss_command_updates_mode(monkeypatch, manifest_store, tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    captured_kwargs = {}

    def builder(**kwargs):
        captured_kwargs.update(kwargs)
        orchestrator = Mock()
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
        orchestrator.run_cycle.return_value = OrchestrationResult(
            selected_session="session-main",
            sessions_summary={"main": {"selected": True}},
            artifact=artifact,
        )
        return orchestrator

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=builder,
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )
    controller._config.logs_root = tmp_path / "logs"

    assert controller._config.boss_mode == BossMode.SCORE
    _run_async(controller.handle_input("/boss rewrite"))
    assert controller._config.boss_mode == BossMode.REWRITE
    _run_async(controller.handle_input("Implement feature"))
    assert captured_kwargs["boss_mode"] == BossMode.REWRITE
    settings_path = tmp_path / ".parallel-dev" / "settings.json"
    assert json.loads(settings_path.read_text(encoding="utf-8"))["boss_mode"] == "rewrite"


def test_boss_command_reports_current_mode(tmp_path, caplog):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(event_handler=handler, worktree_root=tmp_path)
    _run_async(controller.handle_input("/boss"))
    log_messages = [payload.get("text", "") for event, payload in events if event == "log"]
    assert any("Boss" in msg and "score" in msg for msg in log_messages if isinstance(msg, str))


def test_boss_command_rejects_invalid_option(tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    class DummyOrchestrator:
        def __init__(self):
            self._tmux = Mock()
            self._worktree = Mock()

        def run_cycle(self, instruction, selector, resume_session_id=None):
            return OrchestrationResult(selected_session="session-x", sessions_summary={})

        def set_main_session_hook(self, hook):  # pragma: no cover - trivial
            self._hook = hook

        def set_worker_decider(self, decider):  # pragma: no cover - trivial
            self._decider = decider

    controller = CLIController(
        event_handler=handler,
        worktree_root=tmp_path,
        orchestrator_builder=lambda **_: DummyOrchestrator(),
    )
    _run_async(controller.handle_input("/boss invalid"))
    log_messages = [payload.get("text", "") for event, payload in events if event == "log"]
    assert not any("使い方" in msg for msg in log_messages)


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


def test_command_suggestions_and_options(manifest_store, tmp_path):
    controller = CLIController(
        event_handler=lambda *_: None,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=manifest_store,
        worktree_root=tmp_path,
    )

    suggestions = controller.get_command_suggestions("/")
    assert any(s.name == "/attach" for s in suggestions)
    assert any(s.name == "/resume" for s in suggestions)

    # Prepare a session manifest to surface resume options
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    conversation = logs_dir / "conversation.jsonl"
    conversation.write_text("{}\n", encoding="utf-8")
    manifest = SessionManifest(
        session_id="session-test",
        created_at="2025-11-02T10:00:00",
        tmux_session="tmux-test",
        worker_count=1,
        mode="parallel",
        logs_dir=str(logs_dir),
        latest_instruction="Test instruction",
        scoreboard={},
        conversation_log=str(conversation),
        selected_session_id=None,
        main=PaneRecord(role="main", name="main", session_id="main-session"),
        boss=None,
        workers={},
    )
    manifest_store.save_manifest(manifest)
    controller._ensure_tmux_session = Mock()

    options = controller.get_command_options("/resume")
    assert options
    index_value = options[0].value
    events: list = []

    def handler(event_type, payload):
        if event_type == "log":
            events.append(payload)

    controller._event_handler = handler
    _run_async(controller.execute_command("/resume", index_value))
    assert any("読み込みました" in payload.get("text", "") for payload in events)
    controller._ensure_tmux_session.assert_called_once()
