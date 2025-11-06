from pathlib import Path
from unittest.mock import Mock

import pytest

from parallel_developer.services import TmuxLayoutManager


class DummyPane:
    def __init__(self, pane_id, height=24, width=120):
        self.pane_id = pane_id
        self.height = height
        self.width = width
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
        self.selected_pane = self.panes[0]

    def split_window(self, attach=False):
        pane = DummyPane(f"%{len(self.panes)}")
        self.panes.append(pane)
        return pane

    def select_layout(self, layout):
        self.select_layout_args.append(layout)

    def select_pane(self, target_pane):
        for pane in self.panes:
            if pane.pane_id == target_pane:
                self.selected_pane = pane
                break


class DummySession:
    def __init__(self, name):
        self.session_name = name
        self.windows = [DummyWindow()]
        self.attached_window = self.windows[0]
        self.cmd_calls = []

    def cmd(self, *args):
        self.cmd_calls.append(args)
        return self


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
        session_namespace="session-a",
    )

    layout = manager.ensure_layout(session_name="parallel-dev", worker_count=2)
    assert layout["main"] == "%0"
    assert layout["boss"] == "%1"
    assert layout["workers"] == ["%2", "%3"]

    session_cmds = monkeypatch_server.sessions[0].cmd_calls
    assert ("set-option", "-g", "mouse", "on") in session_cmds
    assert ("set-option", "-g", "pane-border-style", "fg=green") in session_cmds
    assert ("set-option", "-g", "pane-active-border-style", "fg=orange") in session_cmds
    assert ("set-option", "-g", "pane-border-format", "#{pane_title}") in session_cmds
    assert ("select-pane", "-t", "%0", "-T", "MAIN") in session_cmds
    assert ("select-pane", "-t", "%1", "-T", "BOSS") in session_cmds
    assert ("select-pane", "-t", "%2", "-T", "WORKER-1") in session_cmds
    assert ("select-pane", "-t", "%3", "-T", "WORKER-2") in session_cmds

    manager.launch_main_session(pane_id=layout["main"])
    manager.launch_boss_session(pane_id=layout["boss"])
    manager.send_instruction_to_pane(pane_id=layout["main"], instruction="echo main")
    manager.interrupt_pane(pane_id=layout["main"])

    initial_map = {layout["workers"][0]: "session-worker-1"}
    manager.send_instruction_to_workers(fork_map=initial_map, instruction="echo worker")

    worker_paths = {
        layout["workers"][0]: Path("/repo/.parallel-dev/sessions/session-a/worktrees/worker-1"),
        layout["workers"][1]: Path("/repo/.parallel-dev/sessions/session-a/worktrees/worker-2"),
    }

    fork_list = manager.fork_workers(
        workers=layout["workers"],
        base_session_id="session-main",
        pane_paths=worker_paths,
    )
    assert fork_list == ["%2", "%3"]

    fork_map = {
        layout["workers"][0]: "session-worker-1",
        layout["workers"][1]: "session-worker-2",
    }
    manager.prepare_for_instruction(pane_id=layout["boss"])
    manager.promote_to_main(session_id="session-worker-1", pane_id=layout["main"])

    main_pane = monkeypatch_server.sessions[0].windows[0].panes[0]
    assert any("cd /repo && codex" in entry[0] for entry in main_pane.sent)
    assert any("echo main" in entry[0] for entry in main_pane.sent)
    assert main_pane.sent.count(("C-c", False)) >= 2
    worker_pane = monkeypatch_server.sessions[0].windows[0].panes[2]
    assert any(entry[0].startswith("cd /repo/.parallel-dev/sessions/session-a/worktrees/worker-1 && codex resume") for entry in worker_pane.sent)
    assert worker_pane.sent.count(("C-c", False)) >= 2
    assert worker_pane.sent.count(("C-[", False)) >= 2
    assert worker_pane.sent.count(("", True)) >= 1
    other_worker_pane = monkeypatch_server.sessions[0].windows[0].panes[3]
    assert other_worker_pane.sent.count(("C-[", False)) >= 2
    assert other_worker_pane.sent.count(("", True)) >= 1
    assert main_pane.sent[-1] == ("codex resume session-worker-1", True)

    boss_pane = monkeypatch_server.sessions[0].windows[0].panes[1]
    assert ("C-c", False) in boss_pane.sent
    manager.resume_session(pane_id=layout["main"], workdir=Path("/repo"), session_id="session-main")
    assert any("codex resume session-main" in entry[0] for entry in main_pane.sent)


def test_tmux_layout_manager_recreates_existing_session(monkeypatch_server):
    monitor = Mock()
    manager = TmuxLayoutManager(
        session_name="parallel-dev",
        worker_count=1,
        monitor=monitor,
        root_path=Path("/repo"),
        session_namespace="session-a",
    )

    # Simulate pre-existing session
    existing = DummySession("parallel-dev")
    monkeypatch_server.sessions.append(existing)

    manager.ensure_layout(session_name="parallel-dev", worker_count=1)

    entries = [args for args in monkeypatch_server.new_session_args if args[0] == "parallel-dev"]
    assert entries[0] == ("parallel-dev", False, True)


def test_fork_boss_interrupts_before_resume(monkeypatch_server):
    monitor = Mock()
    manager = TmuxLayoutManager(
        session_name="parallel-dev",
        worker_count=1,
        monitor=monitor,
        root_path=Path("/repo"),
        startup_delay=0.0,
        backtrack_delay=0.0,
        session_namespace="session-a",
    )

    layout = manager.ensure_layout(session_name="parallel-dev", worker_count=1)
    boss_pane = monkeypatch_server.sessions[0].windows[0].panes[1]

    manager.fork_boss(
        pane_id=layout["boss"],
        base_session_id="session-main",
        boss_path=Path("/repo/.parallel-dev/sessions/session-a/worktrees/boss"),
    )

    assert boss_pane.sent[:2] == [("C-c", False), ("C-c", False)]
    assert any(
        entry[0].startswith("cd /repo/.parallel-dev/sessions/session-a/worktrees/boss && codex resume session-main")
        for entry in boss_pane.sent
    )
