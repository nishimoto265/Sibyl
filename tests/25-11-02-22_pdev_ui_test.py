import pytest

from parallel_developer.cli import CommandPalette, ParallelDeveloperApp


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
