import pytest

from textual import events
from textual.widgets import OptionList

from parallel_developer.cli import ParallelDeveloperApp
from parallel_developer.ui.widgets import CommandPalette, ControllerEvent, EventLog, CommandTextArea


@pytest.mark.asyncio
async def test_command_palette_opens_on_slash() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.press("/")  # 入力欄にスラッシュを送信
        await pilot.pause()
        palette = app.query_one("#command-palette", CommandPalette)
        assert palette.display is True
        active = palette.get_active_item()
        assert active is not None
        assert active.label  # ラベルが空でないことを確認


@pytest.mark.asyncio
async def test_command_palette_navigate_with_arrow() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.press("/")
        await pilot.pause()
        palette = app.query_one("#command-palette", CommandPalette)
        first = palette.get_active_item()
        await pilot.press("down")
        await pilot.pause()
        second = palette.get_active_item()
        assert first is not None
        assert second is not None
        assert second.value != first.value


@pytest.mark.asyncio
async def test_command_palette_enter_handles_selection() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.press("/")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        palette = app.query_one("#command-palette", CommandPalette)
        assert palette.display is True  # optionsリストが開いていること
        active = palette.get_active_item()
        assert app._pending_command == "/attach"
        assert active is not None and str(active.value) in {"auto", "manual", "now"}


@pytest.mark.asyncio
async def test_click_log_focuses_log_for_selection() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        event_log = app.query_one("#log", EventLog)
        await pilot.click("#log")
        await pilot.pause()
        assert event_log.has_focus


@pytest.mark.asyncio
async def test_click_body_refocuses_input_when_selection_hidden() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        command_input = app.query_one("#command", CommandTextArea)
        await pilot.click("#body")
        await pilot.pause()
        assert command_input.has_focus


@pytest.mark.asyncio
async def test_option_list_click_keeps_focus_on_selection() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        payload = {
            "candidates": ["1. worker-1 (session abc)", "2. worker-2 (session def)"],
            "scoreboard": {},
        }
        app.on_controller_event(ControllerEvent("selection_request", payload))
        await pilot.pause()
        selection = app.query_one("#selection", OptionList)
        await pilot.click("#selection")
        await pilot.pause()
        assert selection.has_focus


@pytest.mark.asyncio
async def test_event_log_copy_to_clipboard() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        event_log = app.query_one("#log", EventLog)
        event_log.log("alpha")
        event_log.log("beta")
        await pilot.pause()
        select_event = events.Key("ctrl+a", None)
        copy_event = events.Key("ctrl+alt+c", None)
        assert app._handle_text_shortcuts(select_event) is True
        selection = event_log.text_selection
        if selection:
            extracted = event_log.get_selection(selection)
            if extracted:
                text, ending = extracted
                assert "alpha" in text
                assert "beta" in text
        app.copy_to_clipboard("")
        assert app._handle_text_shortcuts(copy_event) is True
        await pilot.pause()
        assert "alpha" in app.clipboard
        assert "beta" in app.clipboard


@pytest.mark.asyncio
async def test_log_command_copy() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        log_widget = app.query_one("#log", EventLog)
        log_widget.log("alpha")
        log_widget.log("beta")
        await pilot.pause()
        await app.controller.execute_command("/log", "copy")


@pytest.mark.asyncio
async def test_ctrl_c_single_press_clears_input() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        command_input = app.query_one("#command", CommandTextArea)
        command_input.text = "hello"
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert command_input.text == ""
        assert app._ctrl_c_armed is True
        assert app.is_running


@pytest.mark.asyncio
async def test_ctrl_c_double_press_exits() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert app.is_running
        await pilot.press("ctrl+c")
        await pilot.pause()
        assert not app.is_running


@pytest.mark.asyncio
async def test_slash_typing_preserves_order() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.press("/")
        await pilot.press("s")
        await pilot.press("a")
        await pilot.pause()
        command_input = app.query_one("#command", CommandTextArea)
        assert command_input.text == "/sa"


@pytest.mark.asyncio
async def test_command_palette_arrow_moves_once() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        await pilot.press("/")
        await pilot.pause()
        palette = app.query_one("#command-palette", CommandPalette)
        initial_index = palette._active_index
        await pilot.press("down")
        await pilot.pause()
        assert palette._active_index == initial_index + 1
        await pilot.press("up")
        await pilot.pause()
        assert palette._active_index == initial_index


@pytest.mark.asyncio
async def test_log_command_save(tmp_path) -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        log_widget = app.query_one("#log", EventLog)
        log_widget.log("alpha")
        log_widget.log("beta")
        await pilot.pause()
        dest = tmp_path / "out.log"
        await app.controller.execute_command("/log", f"save {dest}")
        await pilot.pause()
        assert dest.read_text(encoding="utf-8").strip().splitlines() == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_shift_enter_inserts_newline() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        assert app.command_input is not None
        submitted: list[str] = []

        original_submit = app._submit_command_input

        def capture_submit() -> None:
            if app.command_input:
                submitted.append(app.command_input.text.rstrip("\n"))
                app._set_command_text("")

        app._submit_command_input = capture_submit  # type: ignore[assignment]
        try:
            app.command_input.insert("line1")
            await pilot.pause()
            await pilot.press("shift+enter")
            await pilot.pause()
            app.command_input.insert("line2")
            await pilot.pause()
            assert app.command_input.text == "line1\nline2"
            assert submitted == []
        finally:
            app._submit_command_input = original_submit  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_plain_enter_triggers_submission() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        assert app.command_input is not None
        submitted: list[str] = []

        original_submit = app._submit_command_input

        def capture_submit() -> None:
            if app.command_input:
                submitted.append(app.command_input.text.rstrip("\n"))
                app._set_command_text("")

        app._submit_command_input = capture_submit  # type: ignore[assignment]
        try:
            app.command_input.insert("message")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert submitted == ["message"]
        finally:
            app._submit_command_input = original_submit  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_shift_then_enter_inserts_newline() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        assert app.command_input is not None
        submitted: list[str] = []

        original_submit = app._submit_command_input

        def capture_submit() -> None:
            if app.command_input:
                submitted.append(app.command_input.text.rstrip("\n"))
                app._set_command_text("")

        app._submit_command_input = capture_submit  # type: ignore[assignment]
        try:
            app.command_input.insert("before")
            await pilot.pause()
            await pilot.press("shift")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.command_input.text.endswith("before\n")
            assert submitted == []
        finally:
            app._submit_command_input = original_submit  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_escape_broadcasts_to_tmux(monkeypatch) -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        app.controller.broadcast_escape = lambda: None  # type: ignore[assignment]

        await pilot.press("escape")
        await pilot.pause()

        assert app.controller._paused is True
        assert app.command_input.has_class("paused")
        assert app.command_input.placeholder.startswith("一時停止中")

@pytest.mark.asyncio
async def test_shift_enter_combo_inserts_newline() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        assert app.command_input is not None
        submitted: list[str] = []

        original_submit = app._submit_command_input

        def capture_submit() -> None:
            if app.command_input:
                submitted.append(app.command_input.text.rstrip("\n"))
                app._set_command_text("")

        app._submit_command_input = capture_submit  # type: ignore[assignment]
        try:
            app.command_input.insert("before")
            await pilot.pause()
            await pilot.press("shift+enter")
            await pilot.pause()
            assert app.command_input.text.endswith("before\n")
            assert submitted == []
        finally:
            app._submit_command_input = original_submit  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_shift_enter_plain_enter_only_regression() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        assert app.command_input is not None
        submitted: list[str] = []

        original_submit = app._submit_command_input

        def capture_submit() -> None:
            if app.command_input:
                submitted.append(app.command_input.text.rstrip("\n"))
                app._set_command_text("")

        app._submit_command_input = capture_submit  # type: ignore[assignment]
        try:
            app.command_input.insert("before")
            await pilot.pause()
            await pilot.press("shift")
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert submitted == []
            assert app.command_input.text.endswith("before\n")
            await pilot.press("enter")
            await pilot.pause()
            assert submitted == ["before"]
            assert app.command_input.text == ""
        finally:
            app._submit_command_input = original_submit  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_ctrl_enter_submits_instruction() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        await pilot.pause()
        assert app.command_input is not None
        submitted: list[str] = []

        original_submit = app._submit_command_input

        def capture_submit() -> None:
            if app.command_input:
                submitted.append(app.command_input.text)
                app._set_command_text("")

        app._submit_command_input = capture_submit  # type: ignore[assignment]
        try:
            app.command_input.insert("message")
            await pilot.pause()
            await pilot.press("ctrl+enter")
            await pilot.pause()
            assert submitted == ["message"]
        finally:
            app._submit_command_input = original_submit  # type: ignore[assignment]
