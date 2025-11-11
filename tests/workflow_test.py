import asyncio
from pathlib import Path
from unittest.mock import Mock

import pytest

from parallel_developer.controller import CLIController, FlowMode
from parallel_developer.orchestrator import CandidateInfo, CycleArtifact, CycleLayout, OrchestrationResult, BossMode
from parallel_developer.stores import ManifestStore


def _run(coro):
    return asyncio.run(coro)


class DummyOrchestrator:
    def __init__(self, controller, result, during_run=None):
        self._controller = controller
        self._result = result
        self._during_run = during_run
        self._tmux = Mock()

    def run_cycle(self, instruction, selector, resume_session_id=None):
        if self._during_run:
            self._during_run()
        return self._result

    def set_main_session_hook(self, hook):
        self._main_hook = hook

    def set_worker_decider(self, decider):
        self._worker_decider = decider


@pytest.fixture
def base_controller(tmp_path):
    events = []

    def handler(event_type, payload):
        events.append((event_type, payload))

    controller = CLIController(
        event_handler=handler,
        orchestrator_builder=lambda **_: Mock(),
        manifest_store=ManifestStore(tmp_path / "manifests"),
        worktree_root=tmp_path,
    )
    controller._emit = lambda event, payload: events.append((event, payload))
    async def _noop_handle_attach(**kwargs):
        return None

    controller._handle_attach_command = _noop_handle_attach  # type: ignore[attr-defined]
    controller._attach_mode = "manual"
    return controller, events



def test_workflow_cancel_replays_queued(base_controller):
    controller, _ = base_controller

    def during_run():
        controller._cancelled_cycles.add(controller._cycle_counter)

    result_cancel = OrchestrationResult(selected_session="session-a", sessions_summary={})
    result_followup = OrchestrationResult(selected_session="session-b", sessions_summary={"main": {"selected": True}})

    orchestrators = [
        DummyOrchestrator(controller, result_cancel, during_run=during_run),
        DummyOrchestrator(controller, result_followup),
    ]

    def builder(**kwargs):
        return orchestrators.pop(0)

    controller._builder = builder
    controller._queued_instruction = "second pass"

    _run(controller._run_instruction("first pass"))

    assert controller._queued_instruction is None
    assert controller._last_selected_session == "session-b"


def test_cancelled_cycle_resumes_previous_session(base_controller):
    controller, _ = base_controller

    resume_ids = []

    class RecordingOrchestrator:
        def __init__(self, controller, session_id, cancel):
            self._controller = controller
            self._session_id = session_id
            self._cancel = cancel
            self._main_hook = None
            self._worker_decider = None
            self._tmux = Mock()

        def set_main_session_hook(self, hook):
            self._main_hook = hook

        def set_worker_decider(self, decider):
            self._worker_decider = decider

        def run_cycle(self, instruction, selector, resume_session_id=None):
            resume_ids.append(resume_session_id)
            if self._main_hook:
                self._main_hook(self._session_id)
            if self._cancel:
                self._controller._cancelled_cycles.add(self._controller._cycle_counter)
            return OrchestrationResult(
                selected_session=self._session_id,
                sessions_summary={},
            )

    controller._cycle_history = [
        {
            "cycle_id": 1,
            "selected_session": "session-prev",
            "scoreboard": {},
            "instruction": "previous",
        }
    ]
    controller._last_selected_session = "session-prev"
    controller._active_main_session_id = "session-prev"

    orchestrators = [
        RecordingOrchestrator(controller, "session-new", cancel=True),
        RecordingOrchestrator(controller, "session-next", cancel=False),
    ]

    def builder(**kwargs):
        return orchestrators.pop(0)

    controller._builder = builder

    _run(controller._run_instruction("first cancelled"))
    assert resume_ids[0] == "session-prev"
    assert controller._last_selected_session == "session-prev"

    _run(controller._run_instruction("second run"))
    assert resume_ids[1] == "session-prev"


def test_first_cycle_cancel_keeps_current_session(base_controller):
    controller, _ = base_controller

    resume_ids = []

    class RecordingOrchestrator:
        def __init__(self, controller, session_id):
            self._controller = controller
            self._session_id = session_id
            self._main_hook = None
            self._worker_decider = None
            self._tmux = Mock()

        def set_main_session_hook(self, hook):
            self._main_hook = hook

        def set_worker_decider(self, decider):
            self._worker_decider = decider

        def run_cycle(self, instruction, selector, resume_session_id=None):
            resume_ids.append(resume_session_id)
            if self._main_hook:
                self._main_hook(self._session_id)
            self._controller._cancelled_cycles.add(self._controller._cycle_counter)
            return OrchestrationResult(
                selected_session=self._session_id,
                sessions_summary={},
            )

    orchestrators = [RecordingOrchestrator(controller, "session-new"), RecordingOrchestrator(controller, "session-next")]

    def builder(**kwargs):
        return orchestrators.pop(0)

    controller._builder = builder

    _run(controller._run_instruction("first cancelled"))
    assert resume_ids[0] is None
    assert controller._last_selected_session == "session-new"

    _run(controller._run_instruction("second run"))
    assert resume_ids[1] == "session-new"


def test_run_instruction_triggers_auto_attach(base_controller):
    controller, _ = base_controller
    controller._attach_mode = "auto"
    attach_calls = []

    async def fake_attach(**kwargs):
        attach_calls.append(kwargs)

    controller._handle_attach_command = fake_attach  # type: ignore[assignment]

    result = OrchestrationResult(selected_session="session-main", sessions_summary={"main": {"selected": True}})
    controller._builder = lambda **_: DummyOrchestrator(controller, result)

    _run(controller._run_instruction("auto attach check"))

    assert attach_calls == [{"force": False}]


def test_run_instruction_saves_manifest_when_artifact_present(base_controller, tmp_path):
    controller, _ = base_controller
    artifact = CycleArtifact(
        main_session_id="session-main",
        worker_sessions={"worker-1": "session-worker-1"},
        boss_session_id=None,
        worker_paths={"worker-1": tmp_path / "worker-1"},
        boss_path=None,
        instruction="build feature",
        tmux_session="parallel-dev",
    )
    artifact.selected_session_id = "session-worker-1"
    result = OrchestrationResult(
        selected_session="session-worker-1",
        sessions_summary={"worker-1": {"selected": True}},
        artifact=artifact,
    )
    manifest_object = object()
    controller._build_manifest = Mock(return_value=manifest_object)  # type: ignore[assignment]
    controller._manifest_store.save_manifest = Mock()  # type: ignore[assignment]
    controller._builder = lambda **_: DummyOrchestrator(controller, result)

    _run(controller._run_instruction("persist manifest"))

    controller._manifest_store.save_manifest.assert_called_once_with(manifest_object)  # type: ignore[attr-defined]


def test_run_instruction_cancelled_invokes_revert(base_controller):
    controller, _ = base_controller

    def during_run():
        controller._cancelled_cycles.add(controller._cycle_counter)

    controller._perform_revert = Mock()  # type: ignore[assignment]
    result = OrchestrationResult(selected_session="session-main", sessions_summary={})
    controller._builder = lambda **_: DummyOrchestrator(controller, result, during_run=during_run)

    _run(controller._run_instruction("cancel cycle"))

    controller._perform_revert.assert_called_once_with(silent=True)  # type: ignore[attr-defined]


def test_flow_auto_review_skips_worker_prompt(base_controller):
    controller, events = base_controller
    controller._flow_mode = FlowMode.AUTO_REVIEW
    controller._config.flow_mode = FlowMode.AUTO_REVIEW
    events.clear()
    layout = CycleLayout(
        main_pane="%0",
        boss_pane="%1",
        worker_panes=[],
        worker_names=[],
        pane_to_worker={},
        pane_to_path={},
    )
    decision = controller._handle_worker_decision({}, {}, layout)
    assert decision.action == "done"
    assert any("flow auto review" in payload.get("text", "").lower() for event, payload in events if event == "log")


def test_handle_worker_decision_manual_continue(monkeypatch, base_controller):
    controller, events = base_controller
    controller._flow_mode = FlowMode.MANUAL
    controller._config.flow_mode = FlowMode.MANUAL
    layout = CycleLayout(
        main_pane="%0",
        boss_pane="%1",
        worker_panes=["%2"],
        worker_names=["worker-1"],
        pane_to_worker={"%2": "worker-1"},
        pane_to_path={"%2": Path("/tmp/work")},
    )

    monkeypatch.setattr(controller, "_await_worker_command", lambda: "continue")
    monkeypatch.setattr(controller, "_await_continuation_instruction", lambda: "追加タスクを続行")

    decision = controller._handle_worker_decision({}, {}, layout)

    assert decision.action == "continue"
    assert decision.instruction == "追加タスクを続行"


def test_handle_worker_decision_manual_done(monkeypatch, base_controller):
    controller, _ = base_controller
    layout = CycleLayout(
        main_pane="%0",
        boss_pane="%1",
        worker_panes=["%2"],
        worker_names=["worker-1"],
        pane_to_worker={"%2": "worker-1"},
        pane_to_path={"%2": Path("/tmp/work")},
    )

    monkeypatch.setattr(controller, "_await_worker_command", lambda: "done")

    decision = controller._handle_worker_decision({}, {}, layout)

    assert decision.action == "done"


def test_flow_auto_select_picks_highest_score(base_controller, tmp_path):
    controller, events = base_controller
    controller._flow_mode = FlowMode.AUTO_SELECT
    controller._config.flow_mode = FlowMode.AUTO_SELECT
    controller._config.boss_mode = BossMode.SCORE
    events.clear()
    candidates = [
        CandidateInfo(key="worker-1", label="worker-1", session_id="session-1", branch="branch-1", worktree=tmp_path / "w1"),
        CandidateInfo(key="worker-2", label="worker-2", session_id="session-2", branch="branch-2", worktree=tmp_path / "w2"),
    ]
    scoreboard = {
        "worker-1": {"score": 70, "comment": "ok"},
        "worker-2": {"score": 90, "comment": "best"},
    }
    decision = controller._select_candidates(candidates, scoreboard)
    assert decision.selected_key == "worker-2"
    assert any("flow auto_select" in payload.get("text", "").lower() for event, payload in events if event == "log")
    assert controller._selection_context is None


def test_flow_full_auto_prefers_boss_on_rewrite(base_controller, tmp_path):
    controller, events = base_controller
    controller._flow_mode = FlowMode.FULL_AUTO
    controller._config.flow_mode = FlowMode.FULL_AUTO
    controller._config.boss_mode = BossMode.REWRITE
    events.clear()
    candidates = [
        CandidateInfo(key="worker-1", label="worker-1", session_id="session-1", branch="branch-1", worktree=tmp_path / "w1"),
        CandidateInfo(key="boss", label="boss", session_id="boss-session", branch="branch-boss", worktree=tmp_path / "boss"),
    ]
    scoreboard = {
        "worker-1": {"score": 95, "comment": "great"},
        "boss": {"comment": "rewrite ready"},
    }
    decision = controller._select_candidates(candidates, scoreboard)
    assert decision.selected_key == "boss"
    assert any("flow full_auto" in payload.get("text", "").lower() for event, payload in events if event == "log")
