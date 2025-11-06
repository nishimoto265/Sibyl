import json
from pathlib import Path
from typing import List
from unittest.mock import Mock, call

import pytest

from parallel_developer.orchestrator import (
    CandidateInfo,
    BossMode,
    Orchestrator,
    SelectionDecision,
)


@pytest.fixture
def dependencies():
    tmux = Mock(name="tmux_manager")
    worktree = Mock(name="worktree_manager")
    worktree.root = Path("/repo")
    worktree.boss_path = Path("/repo/.parallel-dev/sessions/session-a/worktrees/boss")
    worktree.boss_branch = "parallel-dev/session-a/boss"
    worktree.worker_branch.side_effect = lambda name: f"parallel-dev/session-a/{name}"
    monitor = Mock(name="monitor")
    monitor.consume_session_until_eof = Mock(name="consume_session_until_eof")
    monitor.refresh_session_id = Mock(name="refresh_session_id", side_effect=lambda sid: sid)
    monitor.bind_existing_session = Mock(name="bind_existing_session")
    logger = Mock(name="log_manager")

    worktree.prepare.return_value = {
        "worker-1": Path("/repo/.parallel-dev/sessions/session-a/worktrees/worker-1"),
        "worker-2": Path("/repo/.parallel-dev/sessions/session-a/worktrees/worker-2"),
        "worker-3": Path("/repo/.parallel-dev/sessions/session-a/worktrees/worker-3"),
    }

    worker_panes = ["pane-worker-1", "pane-worker-2", "pane-worker-3"]
    layout = {
        "main": "pane-main",
        "boss": "pane-boss",
        "workers": worker_panes,
    }
    tmux.ensure_layout.return_value = layout

    fork_map = {
        "pane-worker-1": "session-worker-1",
        "pane-worker-2": "session-worker-2",
        "pane-worker-3": "session-worker-3",
    }
    tmux.fork_workers.return_value = worker_panes

    instruction = "Implement feature X"
    monitor.capture_instruction.return_value = "session-main"
    monitor.await_completion.side_effect = [
        {
            "session-worker-1": {"done": True},
            "session-worker-2": {"done": True},
            "session-worker-3": {"done": True},
        },
        {"session-boss": {"done": True}},
    ]

    monitor.snapshot_rollouts.side_effect = [
        {},  # before ensure_layout
        {Path("/rollout-main"): 1.0},
        {Path("/rollout-main"): 1.0},
    ]
    monitor.register_new_rollout.side_effect = [
        "session-main",
        "session-boss",
    ]
    monitor.get_last_assistant_message.return_value = json.dumps(
        {
            "scores": {
                "worker-1": {"score": 75, "comment": "solid"},
                "worker-2": {"score": 60, "comment": "ok"},
                "boss": {"score": 80, "comment": "merged"},
            }
        }
    )
    monitor.register_worker_rollouts.return_value = fork_map
    return {
        "tmux": tmux,
        "worktree": worktree,
        "monitor": monitor,
        "logger": logger,
        "instruction": instruction,
        "fork_map": fork_map,
    }


def test_orchestrator_runs_happy_path(dependencies):
    orchestrator = Orchestrator(
        tmux_manager=dependencies["tmux"],
        worktree_manager=dependencies["worktree"],
        monitor=dependencies["monitor"],
        log_manager=dependencies["logger"],
        worker_count=3,
        session_name="parallel-dev",
        boss_mode=BossMode.SCORE,
    )

    def selector(candidates, scoreboard=None):
        scores = {candidate.key: 0.0 for candidate in candidates}
        scores["boss"] = 1.0
        return SelectionDecision(selected_key="boss", scores=scores)

    result = orchestrator.run_cycle(dependencies["instruction"], selector=selector)

    dependencies["worktree"].prepare.assert_called_once()
    tmux = dependencies["tmux"]
    monitor = dependencies["monitor"]
    worktree = dependencies["worktree"]

    worktree.prepare.assert_called_once()
    tmux.ensure_layout.assert_called_once_with(
        session_name="parallel-dev",
        worker_count=3,
    )
    tmux.launch_main_session.assert_called_once_with(pane_id="pane-main")
    tmux.launch_boss_session.assert_not_called()
    tmux.set_boss_path.assert_called_once_with(Path("/repo/.parallel-dev/sessions/session-a/worktrees/boss"))
    monitor.register_new_rollout.assert_any_call(pane_id="pane-main", baseline={})
    monitor.register_new_rollout.assert_any_call(
        pane_id="pane-boss", baseline={Path("/rollout-main"): 1.0}
    )
    send_calls = tmux.send_instruction_to_pane.call_args_list
    main_call = send_calls[0]
    assert main_call.kwargs["pane_id"] == "pane-main"
    assert main_call.kwargs["instruction"] == "Fork"
    monitor.wait_for_rollout_activity.assert_called_once_with("session-main", timeout_seconds=10.0)
    tmux.interrupt_pane.assert_called_once_with(pane_id="pane-main")
    monitor.capture_instruction.assert_called_once_with(
        pane_id="pane-main",
        instruction=main_call.kwargs["instruction"],
    )
    expected_worker_paths = {
        "pane-worker-1": Path("/repo/.parallel-dev/sessions/session-a/worktrees/worker-1"),
        "pane-worker-2": Path("/repo/.parallel-dev/sessions/session-a/worktrees/worker-2"),
        "pane-worker-3": Path("/repo/.parallel-dev/sessions/session-a/worktrees/worker-3"),
    }
    tmux.fork_workers.assert_called_once_with(
        workers=["pane-worker-1", "pane-worker-2", "pane-worker-3"],
        base_session_id="session-main",
        pane_paths=expected_worker_paths,
    )
    worker_calls = send_calls[1:4]
    assert len(worker_calls) == 3
    for idx, call in enumerate(worker_calls, start=1):
        assert call.kwargs["pane_id"] == f"pane-worker-{idx}"
        message = call.kwargs["instruction"]
        worker_path = expected_worker_paths[f"pane-worker-{idx}"]
        assert str(worker_path) in message
        assert "Run `pwd`" in message
        assert dependencies["instruction"] in message
        assert "/done" in message
    monitor.register_worker_rollouts.assert_called_once()
    tmux.fork_boss.assert_called_once_with(
        pane_id="pane-boss",
        base_session_id="session-main",
        boss_path=Path("/repo/.parallel-dev/sessions/session-a/worktrees/boss"),
    )
    boss_register_call = monitor.register_new_rollout.call_args_list[-1]
    assert boss_register_call.kwargs == {
        "pane_id": "pane-boss",
        "baseline": {Path("/rollout-main"): 1.0},
    }
    prepare_calls = [call.kwargs for call in tmux.prepare_for_instruction.call_args_list]
    assert prepare_calls[:5] == [
        {"pane_id": "pane-main"},
        {"pane_id": "pane-worker-1"},
        {"pane_id": "pane-worker-2"},
        {"pane_id": "pane-worker-3"},
        {"pane_id": "pane-boss"},
    ]
    boss_instruction_call = send_calls[-1]
    assert boss_instruction_call.kwargs["pane_id"] == "pane-boss"
    assert "score" in boss_instruction_call.kwargs["instruction"]
    assert "worker-1" in boss_instruction_call.kwargs["instruction"]
    worker_call, boss_call = monitor.await_completion.call_args_list
    assert worker_call.kwargs == {
        "session_ids": list(dependencies["fork_map"].values())
    }
    assert boss_call.kwargs == {"session_ids": ["session-boss"]}
    worktree.merge_into_main.assert_called_once_with("parallel-dev/session-a/boss")
    tmux.promote_to_main.assert_called_once_with(
        session_id="session-boss",
        pane_id="pane-main",
    )
    monitor.refresh_session_id.assert_called()
    assert monitor.bind_existing_session.call_args_list[-1] == call(
        pane_id="pane-main",
        session_id="session-boss",
    )
    monitor.bind_existing_session.assert_called_with(pane_id="pane-main", session_id="session-boss")
    assert call("session-boss") in monitor.consume_session_until_eof.call_args_list
    dependencies["logger"].record_cycle.assert_called_once()
    assert result.selected_session == "session-boss"
    assert result.sessions_summary["boss"]["score"] == 80.0
    assert result.sessions_summary["boss"]["selected"] is True
    assert result.artifact is not None
    assert result.artifact.main_session_id == "session-main"
    assert result.artifact.worker_sessions == {
        "worker-1": "session-worker-1",
        "worker-2": "session-worker-2",
        "worker-3": "session-worker-3",
    }
    assert result.artifact.boss_session_id == "session-boss"


def test_orchestrator_continue_skips_boss_phase(dependencies):
    decider_called = []

    def worker_decider(fork_map, completion, layout):
        decider_called.append(True)
        return True

    orchestrator = Orchestrator(
        tmux_manager=dependencies["tmux"],
        worktree_manager=dependencies["worktree"],
        monitor=dependencies["monitor"],
        log_manager=dependencies["logger"],
        worker_count=3,
        session_name="parallel-dev",
        worker_decider=worker_decider,
        boss_mode=BossMode.SCORE,
    )

    orchestrator._run_boss_phase = Mock(name="run_boss_phase")

    result = orchestrator.run_cycle(dependencies["instruction"], selector=lambda *_: SelectionDecision("boss", {}))

    assert decider_called
    orchestrator._run_boss_phase.assert_not_called()
    dependencies["logger"].record_cycle.assert_not_called()
    assert result.continue_requested is True
    assert result.artifact is not None
    assert result.artifact.main_session_id == "session-main"


def test_orchestrator_reuses_main_session_without_resume(dependencies):
    orchestrator = Orchestrator(
        tmux_manager=dependencies["tmux"],
        worktree_manager=dependencies["worktree"],
        monitor=dependencies["monitor"],
        log_manager=dependencies["logger"],
        worker_count=3,
        session_name="parallel-dev",
        boss_mode=BossMode.SCORE,
    )

    def selector(candidates, scoreboard=None):
        scores = {candidate.key: 0.0 for candidate in candidates}
        scores["boss"] = 1.0
        return SelectionDecision(selected_key="boss", scores=scores)

    tmux = dependencies["tmux"]
    monitor = dependencies["monitor"]

    monitor.bind_existing_session = Mock()
    monitor.register_new_rollout.reset_mock()
    monitor.register_new_rollout.side_effect = ["session-boss"]
    monitor.capture_instruction.return_value = "session-prev"

    orchestrator.run_cycle(
        dependencies["instruction"],
        selector=selector,
        resume_session_id="session-prev",
    )

    assert monitor.bind_existing_session.call_args_list == [
        call(pane_id="pane-main", session_id="session-prev"),
        call(pane_id="pane-main", session_id="session-boss"),
    ]
    tmux.launch_main_session.assert_not_called()
    assert not tmux.resume_session.called
    monitor.register_new_rollout.assert_called_once_with(
        pane_id="pane-boss",
        baseline={Path("/rollout-main"): 1.0},
    )
    assert monitor.consume_session_until_eof.call_args_list[-1] == call("session-boss")


def test_ensure_done_directive_always_appends(dependencies):
    orchestrator = Orchestrator(
        tmux_manager=Mock(),
        worktree_manager=Mock(),
        monitor=Mock(),
        log_manager=Mock(),
        worker_count=1,
        session_name="parallel-dev",
        boss_mode=BossMode.SCORE,
    )

    instruction = "作業して完了したら /done"
    ensured = orchestrator._ensure_done_directive(instruction)
    assert "When you have completed the requested task" in ensured
    assert "Run `pwd`" in ensured
    assert ensured.endswith("`/done`.")
    ensured_again = orchestrator._ensure_done_directive(ensured)
    assert ensured_again == ensured


def test_orchestrator_skip_boss_mode(dependencies):
    tmux = dependencies["tmux"]
    monitor = dependencies["monitor"]
    orchestrator = Orchestrator(
        tmux_manager=tmux,
        worktree_manager=dependencies["worktree"],
        monitor=monitor,
        log_manager=dependencies["logger"],
        worker_count=3,
        session_name="parallel-dev",
        boss_mode=BossMode.SKIP,
    )

    def selector(candidates, scoreboard=None):
        return SelectionDecision(
            selected_key=candidates[0].key,
            scores={candidate.key: (1.0 if candidate == candidates[0] else 0.0) for candidate in candidates},
        )

    result = orchestrator.run_cycle(dependencies["instruction"], selector=selector)

    tmux.fork_boss.assert_not_called()
    boss_calls = [call for call in monitor.register_new_rollout.call_args_list if call.kwargs.get("pane_id") == "pane-boss"]
    assert not boss_calls
    assert "boss" not in result.sessions_summary


def test_boss_instruction_rewrite_mode():
    tmux = Mock()
    worktree = Mock()
    worktree.root = Path("/repo")
    worktree.boss_path = Path("/repo/.parallel-dev/sessions/session-a/worktrees/boss")
    worktree.boss_branch = "parallel-dev/session-a/boss"
    worktree.worker_branch.side_effect = lambda name: f"parallel-dev/session-a/{name}"
    monitor = Mock()
    logger = Mock()

    orchestrator = Orchestrator(
        tmux_manager=tmux,
        worktree_manager=worktree,
        monitor=monitor,
        log_manager=logger,
        worker_count=2,
        session_name="parallel-dev",
        boss_mode=BossMode.REWRITE,
    )

    text = orchestrator._build_boss_instruction(["worker-1", "worker-2"], "Implement feature X")
    assert "For each candidate" in text
    assert "After you emit the JSON scoreboard" in text
    assert "Run `pwd`" in text
    assert "respond with /done" in text.lower()


def test_rewrite_mode_sends_followup_prompt(dependencies):
    tmux = dependencies["tmux"]
    monitor = dependencies["monitor"]
    monitor.get_last_assistant_message.side_effect = [
        None,
        json.dumps({
            "scores": {
                "worker-1": {"score": 90, "comment": "great"},
                "worker-2": {"score": 70, "comment": "ok"},
            }
        }),
        json.dumps({
            "scores": {
                "worker-1": {"score": 90, "comment": "great"},
                "worker-2": {"score": 70, "comment": "ok"},
            }
        }),
    ]

    orchestrator = Orchestrator(
        tmux_manager=tmux,
        worktree_manager=dependencies["worktree"],
        monitor=monitor,
        log_manager=dependencies["logger"],
        worker_count=3,
        session_name="parallel-dev",
        boss_mode=BossMode.REWRITE,
    )

    def selector(candidates, scoreboard=None):
        scores = {candidate.key: 0.0 for candidate in candidates}
        scores[candidates[0].key] = 1.0
        return SelectionDecision(selected_key=candidates[0].key, scores=scores)

    orchestrator.run_cycle(dependencies["instruction"], selector=selector)

    assert monitor.refresh_session_id.called
    calls = monitor.consume_session_until_eof.call_args_list
    assert calls[-1] == call("session-worker-1")
    boss_calls = [
        call.kwargs["instruction"]
        for call in tmux.send_instruction_to_pane.call_args_list
        if call.kwargs.get("pane_id") == "pane-boss"
    ]
    assert len(boss_calls) >= 2
    assert "Boss integration phase" in boss_calls[1]
