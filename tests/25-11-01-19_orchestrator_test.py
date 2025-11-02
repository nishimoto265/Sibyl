from pathlib import Path
from typing import List
from unittest.mock import Mock

import pytest

from parallel_developer.orchestrator import (
    CandidateInfo,
    Orchestrator,
    SelectionDecision,
)


@pytest.fixture
def dependencies():
    tmux = Mock(name="tmux_manager")
    worktree = Mock(name="worktree_manager")
    worktree.root = Path("/repo")
    worktree.boss_path = Path("/repo/.parallel-dev/boss")
    worktree.boss_branch = "parallel-dev/boss"
    worktree.worker_branch.side_effect = lambda name: f"parallel-dev/{name}"
    monitor = Mock(name="monitor")
    boss = Mock(name="boss_manager")
    logger = Mock(name="log_manager")

    worktree.prepare.return_value = {
        "worker-1": Path("/repo/.parallel-dev/worktrees/worker-1"),
        "worker-2": Path("/repo/.parallel-dev/worktrees/worker-2"),
        "worker-3": Path("/repo/.parallel-dev/worktrees/worker-3"),
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
    monitor.await_completion.return_value = {
        "session-worker-1": {"done": True},
        "session-worker-2": {"done": True},
        "session-worker-3": {"done": True},
    }

    monitor.snapshot_rollouts.side_effect = [
        {},  # before ensure_layout
        {Path("/rollout-main"): 1.0},
        {Path("/rollout-main"): 1.0},
        {Path("/rollout-main"): 1.0},
    ]
    monitor.register_new_rollout.side_effect = [
        "session-main",
        "session-boss-init",
        "session-boss",
    ]
    monitor.register_worker_rollouts.return_value = fork_map
    boss_scores = {
        "worker-1": {"score": 60},
        "worker-2": {"score": 70},
        "worker-3": {"score": 50},
        "boss": {"score": 95},
    }
    boss.auto_select.return_value = (
        SelectionDecision(
            selected_key="boss",
            scores={"worker-1": 60, "worker-2": 70, "worker-3": 50, "boss": 95},
            comments={"boss": "auto"},
        ),
        boss_scores,
    )

    return {
        "tmux": tmux,
        "worktree": worktree,
        "monitor": monitor,
        "boss": boss,
        "logger": logger,
        "instruction": instruction,
        "fork_map": fork_map,
        "boss_scores": boss_scores,
    }


def test_orchestrator_runs_happy_path(dependencies):
    orchestrator = Orchestrator(
        tmux_manager=dependencies["tmux"],
        worktree_manager=dependencies["worktree"],
        monitor=dependencies["monitor"],
        boss_manager=dependencies["boss"],
        log_manager=dependencies["logger"],
        worker_count=3,
        session_name="parallel-dev",
    )

    result = orchestrator.run_cycle(dependencies["instruction"])

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
    tmux.launch_boss_session.assert_called_once_with(pane_id="pane-boss")
    tmux.set_boss_path.assert_called_once_with(Path("/repo/.parallel-dev/boss"))
    monitor.register_new_rollout.assert_any_call(pane_id="pane-main", baseline={})
    monitor.register_new_rollout.assert_any_call(
        pane_id="pane-boss", baseline={Path("/rollout-main"): 1.0}
    )
    main_instruction = tmux.send_instruction_to_pane.call_args.kwargs["instruction"]
    assert "/done" in main_instruction
    assert tmux.send_instruction_to_pane.call_args.kwargs["pane_id"] == "pane-main"
    tmux.interrupt_pane.assert_called_once_with(pane_id="pane-main")
    monitor.capture_instruction.assert_called_once_with(
        pane_id="pane-main",
        instruction=main_instruction,
    )
    expected_worker_paths = {
        "pane-worker-1": Path("/repo/.parallel-dev/worktrees/worker-1"),
        "pane-worker-2": Path("/repo/.parallel-dev/worktrees/worker-2"),
        "pane-worker-3": Path("/repo/.parallel-dev/worktrees/worker-3"),
    }
    tmux.fork_workers.assert_called_once_with(
        workers=["pane-worker-1", "pane-worker-2", "pane-worker-3"],
        base_session_id="session-main",
        pane_paths=expected_worker_paths,
    )
    monitor.register_worker_rollouts.assert_called_once()
    tmux.confirm_workers.assert_called_once_with(dependencies["fork_map"])
    tmux.fork_boss.assert_called_once_with(
        pane_id="pane-boss",
        base_session_id="session-main",
        boss_path=Path("/repo/.parallel-dev/boss"),
    )
    boss_register_call = monitor.register_new_rollout.call_args_list[-1]
    assert boss_register_call.kwargs == {
        "pane_id": "pane-boss",
        "baseline": {Path("/rollout-main"): 1.0},
    }
    tmux.confirm_boss.assert_called_once_with(pane_id="pane-boss")
    monitor.await_completion.assert_called_once_with(
        session_ids=list(dependencies["fork_map"].values())
    )
    dependencies["boss"].auto_select.assert_called_once()
    worktree.merge_into_main.assert_called_once_with("parallel-dev/boss")
    tmux.promote_to_main.assert_called_once_with(
        session_id="session-boss",
        pane_id="pane-main",
    )
    dependencies["logger"].record_cycle.assert_called_once()
    assert result.selected_session == "session-boss"
    assert result.sessions_summary == dependencies["boss_scores"]
