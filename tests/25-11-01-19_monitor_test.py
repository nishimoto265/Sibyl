import json
import time
from pathlib import Path

import pytest

from parallel_developer.services import CodexMonitor


def test_monitor_registers_and_logs_instruction(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=tmp_path / "codex",
        poll_interval=0.01,
    )

    rollout = tmp_path / "sessions" / "rollout-main.jsonl"
    rollout.write_text("", encoding="utf-8")

    monitor.register_session(
        pane_id="pane-main",
        session_id="session-main",
        rollout_path=rollout,
    )

    session_id = monitor.capture_instruction(pane_id="pane-main", instruction="Build feature")
    assert session_id == "session-main"

    instruction_log = tmp_path / "instruction.log"
    log_entries = [json.loads(line) for line in instruction_log.read_text(encoding="utf-8").splitlines()]
    assert log_entries == [{"pane": "pane-main", "instruction": "Build feature"}]


def test_monitor_waits_for_done(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=tmp_path / "codex",
        poll_interval=0.01,
    )

    rollout_a = tmp_path / "sessions" / "rollout-a.jsonl"
    rollout_b = tmp_path / "sessions" / "rollout-b.jsonl"
    rollout_a.write_text("", encoding="utf-8")
    rollout_b.write_text("", encoding="utf-8")

    monitor.register_session(pane_id="pane-a", session_id="session-a", rollout_path=rollout_a)
    monitor.register_session(pane_id="pane-b", session_id="session-b", rollout_path=rollout_b)

    completion = monitor.await_completion(session_ids=["session-a", "session-b"], timeout_seconds=0.05)
    assert completion["session-a"]["done"] is False
    assert completion["session-b"]["done"] is False

    done_payload_text = {
        "type": "response_item",
        "payload": {
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": '{"scores":{}}\n/done'},
            ],
        },
    }
    done_payload_json = {
        "type": "response_item",
        "payload": {
            "role": "assistant",
            "content": [
                {"type": "output_json", "json": {"scores": {"worker-1": {"score": 100}}}},
            ],
        },
    }
    rollout_a.write_text(json.dumps(done_payload_text) + "\n", encoding="utf-8")
    rollout_b.write_text(json.dumps(done_payload_json) + "\n", encoding="utf-8")

    completion = monitor.await_completion(session_ids=["session-a", "session-b"], timeout_seconds=0.1)
    assert completion["session-a"]["done"] is True
    assert completion["session-b"]["done"] is True


def test_monitor_detects_new_sessions(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    codex_root = tmp_path / "codex"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=codex_root,
        poll_interval=0.01,
    )

    codex_root.mkdir(parents=True, exist_ok=True)
    baseline = monitor.snapshot_rollouts()

    rollout_path = codex_root / "2025" / "11" / "01" / "rollout-test.jsonl"
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    rollout_path.write_text(
        json.dumps({
            "type": "session_meta",
            "payload": {
                "id": "session-worker",
                "timestamp": "2025-11-01T00:00:00Z",
                "cwd": "/repo",
                "originator": "codex_cli_rs",
                "cli_version": "0.46.0",
                "instructions": "",
                "source": "cli",
            },
        })
        + "\n",
        encoding="utf-8",
    )

    session_id = monitor.register_new_rollout(pane_id="pane-worker-1", baseline=baseline, timeout_seconds=0.1)
    assert session_id == "session-worker"

    baseline = monitor.snapshot_rollouts()
    rollout_worker2 = codex_root / "2025" / "11" / "01" / "rollout-worker2.jsonl"
    rollout_worker2.write_text(
        json.dumps({
            "type": "session_meta",
            "payload": {
                "id": "session-worker-2",
                "timestamp": "2025-11-01T00:01:00Z",
                "cwd": "/repo2",
                "originator": "codex_cli_rs",
                "cli_version": "0.46.0",
                "instructions": "",
                "source": "cli",
            },
        })
        + "\n",
        encoding="utf-8",
    )

    mapping = monitor.register_worker_rollouts(
        worker_panes=["pane-worker-2"],
        baseline=baseline,
        timeout_seconds=0.1,
    )

    assert mapping == {"pane-worker-2": "session-worker-2"}


def test_monitor_worker_rollouts_timeout(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    codex_root = tmp_path / "codex"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=codex_root,
        poll_interval=0.01,
    )

    baseline = monitor.snapshot_rollouts()
    with pytest.raises(TimeoutError):
        monitor.register_worker_rollouts(
            worker_panes=["pane-1", "pane-2"],
            baseline=baseline,
            timeout_seconds=0.05,
        )
