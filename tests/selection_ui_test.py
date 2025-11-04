from pathlib import Path

from parallel_developer.cli import ParallelDeveloperApp
from parallel_developer.ui.widgets import ControllerEvent


class DummySelectionList:
    def __init__(self) -> None:
        self.options = []
        self.display = False
        self.focus_called = False

    def clear_options(self) -> None:
        self.options.clear()

    def add_option(self, option) -> None:
        self.options.append(option)

    def focus(self) -> None:
        self.focus_called = True


class DummyInput:
    def __init__(self) -> None:
        self.display = True
        self.focus_called = False

    def focus(self) -> None:
        self.focus_called = True


class DummyLog:
    def __init__(self) -> None:
        self.lines = []

    def log(self, text: str) -> None:
        self.lines.append(text)


class DummyStatus:
    def update_status(self, config, message: str) -> None:
        pass


def test_selection_request_flow(tmp_path: Path):
    app = ParallelDeveloperApp()
    app.selection_list = DummySelectionList()
    app.command_input = DummyInput()
    app.log_panel = DummyLog()
    app.status_panel = DummyStatus()

    payload = {
        "candidates": [
            "1. worker-1 (session abc)",
            "2. worker-2 (session def)",
            "3. boss (session ghi)",
        ],
        "scoreboard": {
            "worker-1": {"score": 80.0, "comment": "ok", "selected": False},
            "worker-2": {"score": 90.0, "comment": "good", "selected": False},
            "boss": {"score": None, "comment": "", "selected": False},
        },
    }

    app.on_controller_event(ControllerEvent("selection_request", payload))

    assert app.selection_list.display is True
    assert len(app.selection_list.options) == 3
    assert app.selection_list.focus_called is True
    assert app.command_input.display is False

    app.on_controller_event(ControllerEvent("selection_finished", {}))

    assert app.selection_list.display is False
    assert app.command_input.display is True
    assert app.command_input.focus_called is True
