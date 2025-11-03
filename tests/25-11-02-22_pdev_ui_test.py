import pytest

from textual import events
from textual.widgets import OptionList

from parallel_developer.cli import (
    CommandPalette,
    ControllerEvent,
    EventLog,
    CommandTextArea,
    ParallelDeveloperApp,
)


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
        await pilot.pause()
        assert "alpha" in app.clipboard
        assert "beta" in app.clipboard


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

        async def _capture_input(value: str) -> None:
            submitted.append(value)

        original_handle_input = app.controller.handle_input

        async def handle_input_override(value: str):
            await _capture_input(value)
            await original_handle_input(value)

        app.controller.handle_input = handle_input_override  # type: ignore[assignment]
        app.command_input.insert("line1")
        await pilot.pause()
        await pilot.press("shift+enter")
        await pilot.pause()
        app.command_input.insert("line2")
        await pilot.pause()
        assert app.command_input.text == "line1\nline2"
        assert submitted == []
