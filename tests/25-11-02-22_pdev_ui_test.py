import pytest

from textual.widgets import Input, OptionList

from parallel_developer.cli import CommandPalette, ControllerEvent, ParallelDeveloperApp


@pytest.mark.asyncio
async def test_command_palette_opens_on_slash() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
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
async def test_click_log_keeps_focus_on_input() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        command_input = app.query_one("#command", Input)
        await pilot.click("#log")
        await pilot.pause()
        assert command_input.has_focus


@pytest.mark.asyncio
async def test_click_body_refocuses_input_when_selection_hidden() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
        command_input = app.query_one("#command", Input)
        await pilot.click("#body")
        await pilot.pause()
        assert command_input.has_focus


@pytest.mark.asyncio
async def test_option_list_click_keeps_focus_on_selection() -> None:
    app = ParallelDeveloperApp()
    async with app.run_test() as pilot:  # type: ignore[attr-defined]
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
