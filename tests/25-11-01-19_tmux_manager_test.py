from pathlib import Path
from unittest.mock import Mock

import pytest

from parallel_developer.services import TmuxLayoutManager


class DummyPane:
    def __init__(self, pane_id):
        self.pane_id = pane_id
        self.sent = []
        self.cmd_calls = []

    def send_keys(self, cmd, enter=True):
        self.sent.append((cmd, enter))

    def cmd(self, *args):
        self.cmd_calls.append(args)


class DummyWindow:
    def __init__(self):
        self.panes = [DummyPane("%0")]
        self.select_layout_args = []

    def split_window(self, attach=False):
        pane = DummyPane(f"%{len(self.panes)}")
        self.panes.append(pane)
        return pane

    def select_layout(self, layout):
        self.select_layout_args.append(layout)


class DummySession:
    def __init__(self, name):
        self.session_name = name
        self.windows = [DummyWindow()]
        self.attached_window = self.windows[0]


class DummyServer:
    def __init__(self):
        self.sessions = []
        self.new_session_args = []

    def find_where(self, attrs):
        for session in self.sessions:
            if session.session_name == attrs.get("session_name"):
                return session
        return None

    def new_session(self, session_name, attach, kill_session=False):
        if kill_session:
            self.sessions = [s for s in self.sessions if s.session_name != session_name]
        session = DummySession(session_name)
        self.sessions.append(session)
        self.new_session_args.append((session_name, attach, kill_session))
        return session


@pytest.fixture
def monkeypatch_server(monkeypatch):
    server = DummyServer()
    monkeypatch.setattr("parallel_developer.services.libtmux.Server", lambda: server)
    return server


def test_tmux_layout_manager_allocates_panes(monkeypatch_server):
    monitor = Mock()

    manager = TmuxLayoutManager(
        session_name="parallel-dev",
        worker_count=2,
        monitor=monitor,
        root_path=Path("/repo"),
    )

    layout = manager.ensure_layout(session_name="parallel-dev", worker_count=2)
    assert layout["main"] == "%0"
    assert layout["boss"] == "%1"
    assert layout["workers"] == ["%2", "%3"]

    manager.launch_main_session(pane_id=layout["main"])
    manager.launch_boss_session(pane_id=layout["boss"])
    manager.send_instruction_to_pane(pane_id=layout["main"], instruction="echo main")
    manager.interrupt_pane(pane_id=layout["main"])

    initial_map = {layout["workers"][0]: "session-worker-1"}
    manager.send_instruction_to_workers(fork_map=initial_map, instruction="echo worker")

    fork_list = manager.fork_workers(workers=layout["workers"], base_session_id="session-main")
    assert fork_list == ["%2", "%3"]

    worker_paths = {
        layout["workers"][0]: Path("/repo/.parallel-dev/worktrees/worker-1"),
        layout["workers"][1]: Path("/repo/.parallel-dev/worktrees/worker-2"),
    }
    fork_map = {
        layout["workers"][0]: "session-worker-1",
        layout["workers"][1]: "session-worker-2",
    }
    manager.resume_workers(fork_map, worker_paths)
    manager.send_instruction_to_workers(fork_map, "echo worker")
    manager.promote_to_main(session_id="session-worker-1", pane_id=layout["main"])

    main_pane = monkeypatch_server.sessions[0].windows[0].panes[0]
    assert "codex --cd /repo" in main_pane.sent[0][0]
    assert main_pane.sent[1] == ("echo main", True)
    assert ("send-keys", "-t", layout["main"], "Escape") in main_pane.cmd_calls
    worker_pane = monkeypatch_server.sessions[0].windows[0].panes[2]
    assert ("send-keys", "-t", layout["workers"][0], "Escape") in worker_pane.cmd_calls
    assert any("codex resume session-worker-1" in cmd for cmd, _ in worker_pane.sent)
    assert worker_pane.sent[-1] == ("echo worker", True)
    assert main_pane.sent[-1] == ("codex resume session-worker-1", True)


def test_tmux_layout_manager_recreates_existing_session(monkeypatch_server):
    monitor = Mock()
    manager = TmuxLayoutManager(
        session_name="parallel-dev",
        worker_count=1,
        monitor=monitor,
        root_path=Path("/repo"),
    )

    # Simulate pre-existing session
    existing = DummySession("parallel-dev")
    monkeypatch_server.sessions.append(existing)

    manager.ensure_layout(session_name="parallel-dev", worker_count=1)

    entries = [args for args in monkeypatch_server.new_session_args if args[0] == "parallel-dev"]
    assert entries[0] == ("parallel-dev", False, True)
