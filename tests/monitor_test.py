import json
import threading
import time
from pathlib import Path

import pytest
import yaml

from parallel_developer.services import CodexMonitor, SessionReservationError


def test_monitor_registers_and_logs_instruction(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=tmp_path / "codex",
        poll_interval=0.01,
        session_namespace="test-session",
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


def test_monitor_bind_existing_session(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=tmp_path / "codex",
        poll_interval=0.01,
        session_namespace="test-session",
    )

    rollout = tmp_path / "sessions" / "rollout-existing.jsonl"
    rollout.parent.mkdir(parents=True, exist_ok=True)
    rollout.write_text("test", encoding="utf-8")

    monitor.register_session(
        pane_id="pane-old",
        session_id="session-existing",
        rollout_path=rollout,
    )
    monitor.bind_existing_session(pane_id="pane-new", session_id="session-existing")

    mapping = yaml.safe_load(session_map.read_text(encoding="utf-8"))
    assert "pane-old" not in mapping["panes"]
    assert mapping["panes"]["pane-new"]["session_id"] == "session-existing"
    assert mapping["sessions"]["session-existing"]["pane_id"] == "pane-new"


def test_monitor_waits_for_done(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=tmp_path / "codex",
        poll_interval=0.01,
        session_namespace="test-session",
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
    assert completion["session-b"]["done"] is False


def test_monitor_force_completion_during_wait(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=tmp_path / "codex",
        poll_interval=0.01,
        session_namespace="test-session",
    )

    rollout = tmp_path / "sessions" / "rollout-force.jsonl"
    rollout.write_text("", encoding="utf-8")
    monitor.register_session(pane_id="pane-force", session_id="session-force", rollout_path=rollout)

    def trigger_force():
        time.sleep(0.02)
        monitor.force_completion(["session-force"])

    threading.Thread(target=trigger_force, daemon=True).start()
    completion = monitor.await_completion(session_ids=["session-force"], timeout_seconds=0.2)
    assert completion["session-force"]["done"] is True
    assert completion["session-force"].get("forced") is True


def test_register_new_rollout_waits_for_session_meta(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    codex_root = tmp_path / "codex"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=codex_root,
        poll_interval=0.01,
        session_namespace="delayed-main",
    )

    baseline = monitor.snapshot_rollouts()
    rollout_path = codex_root / "2025" / "11" / "11" / "rollout-main.jsonl"

    def create_rollout():
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_path.touch()
        time.sleep(0.05)
        rollout_path.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": "session-main-delayed",
                        "timestamp": "2025-11-11T00:00:00Z",
                        "cwd": "/repo",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    threading.Thread(target=create_rollout, daemon=True).start()
    session_id = monitor.register_new_rollout(pane_id="pane-main", baseline=baseline, timeout_seconds=0.5)
    assert session_id == "session-main-delayed"


def test_register_worker_rollouts_waits_for_session_meta(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    codex_root = tmp_path / "codex"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=codex_root,
        poll_interval=0.01,
        session_namespace="delayed-workers",
    )

    baseline = monitor.snapshot_rollouts()
    rollout_path = codex_root / "2025" / "11" / "11" / "rollout-worker.jsonl"

    def create_worker_rollout():
        rollout_path.parent.mkdir(parents=True, exist_ok=True)
        rollout_path.touch()
        time.sleep(0.05)
        rollout_path.write_text(
            json.dumps(
                {
                    "type": "session_meta",
                    "payload": {
                        "id": "session-worker-delayed",
                        "timestamp": "2025-11-11T00:00:01Z",
                        "cwd": "/repo/worker",
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )

    threading.Thread(target=create_worker_rollout, daemon=True).start()
    mapping = monitor.register_worker_rollouts(
        worker_panes=["pane-worker"],
        baseline=baseline,
        timeout_seconds=0.5,
    )
    assert mapping == {"pane-worker": "session-worker-delayed"}


def test_monitor_detects_new_sessions(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    codex_root = tmp_path / "codex"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=codex_root,
        poll_interval=0.01,
        session_namespace="test-session",
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


def test_monitor_register_new_rollout_skips_reserved_sessions(tmp_path: Path):
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    codex_root = tmp_path / "codex"
    codex_root.mkdir(parents=True, exist_ok=True)

    owner_map = shared_dir / "owner.yaml"
    target_map = shared_dir / "target.yaml"

    owner_monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=owner_map,
        codex_sessions_root=codex_root,
        poll_interval=0.01,
        session_namespace="ns-owner",
    )
    target_monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=target_map,
        codex_sessions_root=codex_root,
        poll_interval=0.01,
        session_namespace="ns-target",
    )

    baseline = target_monitor.snapshot_rollouts()

    conflict_path = codex_root / "2025" / "11" / "09" / "rollout-conflict.jsonl"
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    conflict_path.write_text(
        json.dumps({
            "type": "session_meta",
            "payload": {
                "id": "session-conflict",
                "timestamp": "2025-11-09T03:10:00Z",
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

    owner_monitor.register_session(
        pane_id="pane-owner",
        session_id="session-conflict",
        rollout_path=conflict_path,
    )

    time.sleep(0.01)

    target_path = codex_root / "2025" / "11" / "09" / "rollout-target.jsonl"
    target_path.write_text(
        json.dumps({
            "type": "session_meta",
            "payload": {
                "id": "session-target",
                "timestamp": "2025-11-09T03:11:00Z",
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

    session_id = target_monitor.register_new_rollout(
        pane_id="pane-target",
        baseline=baseline,
        timeout_seconds=0.2,
    )

    assert session_id == "session-target"
    target_mapping = yaml.safe_load(target_map.read_text(encoding="utf-8"))
    assert "session-conflict" not in target_mapping.get("sessions", {})
    assert target_mapping["sessions"]["session-target"]["pane_id"] == "pane-target"


def test_monitor_register_worker_rollouts_skip_reserved_sessions(tmp_path: Path):
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    codex_root = tmp_path / "codex"
    codex_root.mkdir(parents=True, exist_ok=True)

    owner_map = shared_dir / "owner.yaml"
    target_map = shared_dir / "target.yaml"

    owner_monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=owner_map,
        codex_sessions_root=codex_root,
        poll_interval=0.01,
        session_namespace="ns-owner",
    )
    target_monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=target_map,
        codex_sessions_root=codex_root,
        poll_interval=0.01,
        session_namespace="ns-target",
    )

    baseline = target_monitor.snapshot_rollouts()

    conflict_path = codex_root / "2025" / "11" / "09" / "rollout-conflict.jsonl"
    conflict_path.parent.mkdir(parents=True, exist_ok=True)
    conflict_path.write_text(
        json.dumps({
            "type": "session_meta",
            "payload": {
                "id": "session-conflict",
                "timestamp": "2025-11-09T03:20:00Z",
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

    owner_monitor.register_session(
        pane_id="pane-owner",
        session_id="session-conflict",
        rollout_path=conflict_path,
    )

    worker_meta = []
    for index in range(2):
        time.sleep(0.01)
        path = codex_root / "2025" / "11" / "09" / f"rollout-worker-{index}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({
                "type": "session_meta",
                "payload": {
                    "id": f"session-worker-{index}",
                    "timestamp": f"2025-11-09T03:2{index}:00Z",
                    "cwd": f"/repo-{index}",
                    "originator": "codex_cli_rs",
                    "cli_version": "0.46.0",
                    "instructions": "",
                    "source": "cli",
                },
            })
            + "\n",
            encoding="utf-8",
        )
        worker_meta.append((f"pane-{index+1}", f"session-worker-{index}"))

    mapping = target_monitor.register_worker_rollouts(
        worker_panes=[name for name, _ in worker_meta],
        baseline=baseline,
        timeout_seconds=0.5,
    )

    assert mapping == {name: session_id for name, session_id in worker_meta}


def test_monitor_worker_rollouts_timeout(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    codex_root = tmp_path / "codex"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=codex_root,
        poll_interval=0.01,
        session_namespace="test-session",
    )

    baseline = monitor.snapshot_rollouts()
    with pytest.raises(TimeoutError):
        monitor.register_worker_rollouts(
            worker_panes=["pane-1", "pane-2"],
            baseline=baseline,
            timeout_seconds=0.05,
        )


def test_register_session_rejects_reserved_rollout(tmp_path: Path):
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    codex_root = tmp_path / "codex"
    codex_root.mkdir(parents=True, exist_ok=True)
    rollout = codex_root / "2025" / "11" / "09" / "rollout.jsonl"
    rollout.parent.mkdir(parents=True, exist_ok=True)
    rollout.write_text("", encoding="utf-8")

    owner_monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=shared_dir / "owner.yaml",
        codex_sessions_root=codex_root,
        poll_interval=0.01,
        session_namespace="owner",
    )
    target_monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=shared_dir / "target.yaml",
        codex_sessions_root=codex_root,
        poll_interval=0.01,
        session_namespace="target",
    )

    owner_monitor.register_session(
        pane_id="pane-owner",
        session_id="session-reserved",
        rollout_path=rollout,
    )

    with pytest.raises(SessionReservationError):
        target_monitor.register_session(
            pane_id="pane-target",
            session_id="session-reserved",
            rollout_path=rollout,
        )


def test_await_completion_uses_signal_flags(tmp_path: Path):
    session_map = tmp_path / "sessions_map.yaml"
    monitor = CodexMonitor(
        logs_dir=tmp_path,
        session_map_path=session_map,
        codex_sessions_root=tmp_path / "codex",
        poll_interval=0.01,
        session_namespace="signal-test",
    )
    rollout = tmp_path / "sessions" / "rollout-signal.jsonl"
    rollout.parent.mkdir(parents=True, exist_ok=True)
    rollout.write_text("", encoding="utf-8")

    monitor.register_session(pane_id="pane-signal", session_id="session-signal", rollout_path=rollout)

    flag_path = tmp_path / "signals" / "session-signal.done"
    flag_path.parent.mkdir(parents=True, exist_ok=True)
    flag_path.write_text("", encoding="utf-8")

    completion = monitor.await_completion(
        session_ids=["session-signal"],
        timeout_seconds=0.2,
        signal_paths={"session-signal": flag_path},
    )

    assert completion["session-signal"]["done"] is True
    assert "session-signal" not in monitor._active_signal_paths  # type: ignore[attr-defined]
