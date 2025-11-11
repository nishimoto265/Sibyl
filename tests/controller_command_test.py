import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from parallel_developer.controller import CLIController, FlowMode, MergeMode
from parallel_developer.orchestrator import BossMode, CandidateInfo, MergeMode as OrchestratorMergeMode, MergeOutcome, OrchestrationResult
from parallel_developer.stores import PaneRecord, SessionManifest, SessionReference, SettingsStore


def _run(coro):
    return asyncio.run(coro)


class DummyManifestStore:
    def __init__(self, sessions=None, manifest=None):
        self._sessions = sessions or []
        self._manifest = manifest
        self.saved = []

    def list_sessions(self):
        return list(self._sessions)

    def save_manifest(self, manifest):
        self.saved.append(manifest)

    def load_manifest(self, session_id):
        if self._manifest is None:
            raise FileNotFoundError(session_id)
        return self._manifest


@pytest.fixture
def event_recorder():
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    return events, handler


@pytest.fixture
def controller(event_recorder, tmp_path):
    events, handler = event_recorder
    store = DummyManifestStore()
    ctrl = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=store,
        worktree_root=tmp_path,
    )
    ctrl._emit = lambda event, payload: events.append((event, payload))
    ctrl._handle_attach_command = AsyncMock()
    return ctrl


def test_attach_command_updates_mode_and_now(controller, event_recorder):
    events, _ = event_recorder
    _run(controller.execute_command("/attach", "manual"))
    assert controller._attach_mode == "manual"
    assert any("/attach モード" in payload.get("text", "") for event, payload in events if event == "log")

    controller._handle_attach_command.reset_mock()
    _run(controller.execute_command("/attach", "now"))
    controller._handle_attach_command.assert_awaited_once()


def test_parallel_command_valid_and_invalid(controller, event_recorder):
    events, _ = event_recorder
    _run(controller.execute_command("/parallel", "3"))
    assert controller._config.worker_count == 3

    events.clear()
    _run(controller.execute_command("/parallel", "abc"))
    assert any("数字" in payload.get("text", "") for event, payload in events if event == "log")

    events.clear()
    _run(controller.execute_command("/parallel", "0"))
    assert any("1以上" in payload.get("text", "") for event, payload in events if event == "log")


def test_mode_command(controller, event_recorder):
    events, _ = event_recorder
    _run(controller.execute_command("/mode", "main"))
    assert controller._config.mode.value == "main"
    assert any(payload.get("message") == "設定を更新しました。" for event, payload in events if event == "status")

    events.clear()
    _run(controller.execute_command("/mode", "invalid"))
    assert any("使い方" in payload.get("text", "") for event, payload in events if event == "log")


def test_merge_command(controller, event_recorder):
    events, _ = event_recorder
    _run(controller.execute_command("/merge", "auto"))
    assert controller._merge_mode == MergeMode.AUTO
    assert any(
        any(keyword in payload.get("text", "") for keyword in ("マージ戦略を", "マージ戦略は既に"))
        for event, payload in events
        if event == "log"
    )

    events.clear()
    _run(controller.execute_command("/merge"))
    assert any("現在のマージ戦略" in payload.get("text", "") for event, payload in events if event == "log")

    events.clear()
    _run(controller.execute_command("/merge", "invalid"))
    assert any("使い方" in payload.get("text", "") for event, payload in events if event == "log")


def test_handle_merge_outcome_logging(controller, event_recorder):
    events, _ = event_recorder
    outcome = MergeOutcome(
        strategy=OrchestratorMergeMode.MANUAL,
        status="delegate",
        branch="feature",
        reason="manual_user",
    )
    controller._handle_merge_outcome(outcome)
    assert any("manualモード" in payload.get("text", "") for event, payload in events if event == "log")

    events.clear()
    delegate = MergeOutcome(
        strategy=OrchestratorMergeMode.AUTO,
        status="delegate",
        branch="feature",
        reason="agent_auto",
    )
    controller._handle_merge_outcome(delegate)
    assert any("Autoモード" in payload.get("text", "") for event, payload in events if event == "log")


def test_status_and_scoreboard_commands(controller, event_recorder):
    events, _ = event_recorder
    _run(controller.execute_command("/status"))
    assert events[-1][0] == "status"

    events.clear()
    controller._last_scoreboard = {"worker-1": {"score": 80}}
    _run(controller.execute_command("/scoreboard"))
    assert events[-1][0] == "scoreboard"


def test_help_and_exit_commands(controller, event_recorder):
    events, _ = event_recorder
    _run(controller.execute_command("/help"))
    assert any("利用可能なコマンド" in payload.get("text", "") for event, payload in events if event == "log")

    events.clear()
    _run(controller.execute_command("/exit"))
    assert events[-1][0] == "quit"


def test_done_command(controller, event_recorder):
    events, _ = event_recorder
    orchestrator = Mock()
    orchestrator.force_complete_workers.return_value = 2
    controller._active_orchestrator = orchestrator
    _run(controller.execute_command("/done"))
    assert any("完了済み" in payload.get("text", "") for event, payload in events if event == "log")

    events.clear()
    controller._active_orchestrator = None
    _run(controller.execute_command("/done"))
    assert any("現在進行中" in payload.get("text", "") for event, payload in events if event == "log")


def test_continue_command_without_future(controller, event_recorder):
    events, _ = event_recorder
    controller._continue_future = None
    _run(controller.execute_command("/continue"))
    assert any("利用できません" in payload.get("text", "") for event, payload in events if event == "log")


def test_log_command_copy_and_save(controller, event_recorder, tmp_path, monkeypatch):
    events, _ = event_recorder

    _run(controller.execute_command("/log", "copy"))
    assert events[-1][0] == "log_copy"

    events.clear()
    dest = tmp_path / "out.log"
    _run(controller.execute_command("/log", f"save {dest}"))
    assert events[-1] == ("log_save", {"path": str(dest)})

    events.clear()
    _run(controller.execute_command("/log", "save"))
    assert any("保存先" in payload.get("text", "") for event, payload in events if event == "log")


def test_prune_boss_candidates_for_non_rewrite(controller):
    controller._config.boss_mode = BossMode.SCORE
    candidates = [
        CandidateInfo(key="worker-1", label="worker-1", session_id="s1", branch="b1", worktree=Path("/tmp/w1")),
        CandidateInfo(key="boss", label="boss", session_id="sb", branch="bb", worktree=Path("/tmp/b")),
    ]
    scoreboard = {
        "worker-1": {"score": 80, "comment": "ok"},
        "boss": {"score": 95, "comment": "rewrite"},
    }

    filtered, filtered_scoreboard = controller._prune_boss_candidates(candidates, scoreboard)

    assert all(candidate.key != "boss" for candidate in filtered)
    assert filtered_scoreboard is not None and "boss" not in filtered_scoreboard


def test_resume_command_lists_and_loads(tmp_path, event_recorder):
    events, handler = event_recorder
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    ref = SessionReference(
        session_id="session-1",
        tmux_session="tmux-1",
        manifest_path=manifest_dir / "session-1.yaml",
        created_at="2025-11-01T00:00:00",
        worker_count=2,
        mode="parallel",
        latest_instruction="Build feature",
        logs_dir=Path("logs/1"),
    )
    manifest = SessionManifest(
        session_id="session-1",
        created_at="2025-11-01T00:00:00",
        tmux_session="tmux-1",
        worker_count=2,
        mode="parallel",
        logs_dir="logs",
        main=PaneRecord(role="main", name=None, session_id="main-session", worktree="/repo"),
        workers={},
        selected_session_id="main-session",
    )
    store = DummyManifestStore(sessions=[ref], manifest=manifest)
    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=store,
        worktree_root=tmp_path,
    )
    controller._ensure_tmux_session = lambda manifest: None
    events.clear()
    _run(controller.execute_command("/resume"))
    assert any("保存されたセッション" in payload.get("text", "") for event, payload in events if event == "log")

    events.clear()
    _run(controller.execute_command("/resume", 1))
    assert controller._last_selected_session == "main-session"


def test_resume_invalid_selection(controller, event_recorder):
    events, _ = event_recorder
    controller._resume_options = []
    _run(controller.execute_command("/resume", 1))
    assert any("先に /resume" in payload.get("text", "") for event, payload in events if event == "log")


def test_flow_command_updates_mode(controller, event_recorder):
    events, _ = event_recorder
    _run(controller.execute_command("/flow"))
    assert any("フローモード" in payload.get("text", "") for event, payload in events if event == "log")

    events.clear()
    _run(controller.execute_command("/flow", "auto_select"))
    assert controller._flow_mode == FlowMode.AUTO_SELECT
    assert controller._config.flow_mode == FlowMode.AUTO_SELECT
    assert controller._settings_store.flow == FlowMode.AUTO_SELECT.value
    assert any("フローモードを" in payload.get("text", "") for event, payload in events if event == "log")

    events.clear()
    _run(controller.execute_command("/flow", "invalid"))
    assert any("使い方" in payload.get("text", "") for event, payload in events if event == "log")


def test_settings_store_loads_and_saves_yaml(tmp_path):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "commands": {
                    "attach": "manual",
                    "boss": "rewrite",
                    "flow": "auto_review",
                    "parallel": "4",
                    "mode": "main",
                    "commit": "auto",
                },
                "paths": {"worktree_root": "/repo/work"},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    store = SettingsStore(cfg)
    assert store.attach == "manual"
    assert store.boss == "rewrite"
    assert store.worktree_root == "/repo/work"

    store.parallel = "5"
    data = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert data["commands"]["parallel"] == "5"


def test_settings_store_update_snapshot(tmp_path):
    cfg = tmp_path / "settings.yaml"
    store = SettingsStore(cfg)
    store.update(attach="manual", worktree_root="/workspace")
    snapshot = store.snapshot()
    assert snapshot["commands"]["attach"] == "manual"
    assert snapshot["paths"]["worktree_root"] == "/workspace"
    assert cfg.read_text(encoding="utf-8")


def test_settings_store_legacy_keys(tmp_path):
    cfg = tmp_path / "legacy.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "attach_mode": "manual",
                "boss_mode": "rewrite",
                "flow_mode": "auto_select",
                "worker_count": "2",
                "session_mode": "main",
                "auto_commit": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    store = SettingsStore(cfg)
    assert store.attach == "manual"
    assert store.boss == "rewrite"
    assert store.flow == "auto_select"
    assert store.parallel == "2"
    assert store.mode == "main"
    assert store.commit == "auto"
