"""UI widgets extracted from the parallel developer CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, TYPE_CHECKING

from rich.text import Text
from textual import events
from textual.message import Message
from textual.widget import Widget
from textual.widgets import OptionList, RichLog, Static, TextArea

if TYPE_CHECKING:
    from parallel_developer.controller import SessionConfig

__all__ = [
    "PaletteItem",
    "ControllerEvent",
    "StatusPanel",
    "EventLog",
    "CommandTextArea",
    "CommandHint",
    "CommandPalette",
]


@dataclass
class PaletteItem:
    label: str
    value: object


class ControllerEvent(Message):
    def __init__(self, event_type: str, payload: Optional[Dict[str, object]] = None) -> None:
        super().__init__()
        self.event_type = event_type
        self.payload = payload or {}


class StatusPanel(Static):
    def update_status(self, config: "SessionConfig", message: str) -> None:
        flow_value = getattr(config, "flow_mode", None)
        if hasattr(flow_value, "value"):
            flow_text = flow_value.value  # type: ignore[attr-defined]
        elif flow_value is not None:
            flow_text = str(flow_value)
        else:
            flow_text = "manual"
        lines = [
            f"tmux session : {config.tmux_session}",
            f"mode         : {config.mode.value}",
            f"flow         : {flow_text}",
            f"workers      : {config.worker_count}",
            f"logs root    : {config.logs_root}",
            f"status       : {message}",
        ]
        self.update("\n".join(lines))


class EventLog(RichLog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, highlight=True, markup=True, **kwargs)
        self.wrap = True
        self.auto_scroll = True
        self.min_width = 0
        self._entries: List[str] = []

    def log(self, text: str) -> None:
        for line in text.splitlines():
            self._entries.append(line)
            self._write_line(line)

    def _write_line(self, line: str) -> None:
        if self.markup:
            renderable = Text.from_markup(line)
        else:
            renderable = Text(line)
        renderable.no_wrap = False
        renderable.overflow = "fold"
        super().write(renderable)

    def on_resize(self, event: events.Resize) -> None:
        super().on_resize(event)
        self._redraw()

    def _redraw(self) -> None:
        if not self._entries:
            return
        if not getattr(self, "_size_known", False):
            return
        super().clear()
        for line in self._entries:
            self._write_line(line)

    @property
    def entries(self) -> List[str]:
        return list(self._entries)


class CommandTextArea(TextArea):
    async def _on_key(self, event: events.Key) -> None:  # type: ignore[override]
        key = event.key or ""
        name = event.name or ""
        aliases = set(event.aliases)

        if not hasattr(self, "_shift_next_enter"):
            self._shift_next_enter = False  # type: ignore[attr-defined]

        if key == "shift":
            event.stop()
            event.prevent_default()
            self._shift_next_enter = True  # type: ignore[attr-defined]
            return

        if key in {"ctrl+enter", "meta+enter"} or name in {"ctrl_enter", "meta_enter"} or aliases.intersection({"ctrl+enter", "meta+enter"}):
            event.stop()
            event.prevent_default()
            self._shift_next_enter = False  # type: ignore[attr-defined]
            app = self.app
            if hasattr(app, "_submit_command_input"):
                app._submit_command_input()  # type: ignore[attr-defined]
            return

        if key == "shift+enter" or name == "shift_enter" or "shift+enter" in aliases:
            event.stop()
            event.prevent_default()
            self._shift_next_enter = False  # type: ignore[attr-defined]
            self.insert("\n")
            return

        if key == "tab" or name == "tab" or "tab" in aliases:
            event.stop()
            event.prevent_default()
            self._shift_next_enter = False  # type: ignore[attr-defined]
            handler = getattr(self.app, "_handle_tab_navigation", None)
            if callable(handler):
                handler(reverse=False)
            return

        if key == "shift+tab" or name == "shift_tab" or "shift+tab" in aliases:
            event.stop()
            event.prevent_default()
            self._shift_next_enter = False  # type: ignore[attr-defined]
            handler = getattr(self.app, "_handle_tab_navigation", None)
            if callable(handler):
                handler(reverse=True)
            return

        if key == "enter":
            event.stop()
            event.prevent_default()
            if getattr(self, "_shift_next_enter", False):
                self._shift_next_enter = False  # type: ignore[attr-defined]
                self.insert("\n")
            else:
                app = self.app
                if hasattr(app, "_submit_command_input"):
                    app._submit_command_input()  # type: ignore[attr-defined]
            return

        self._shift_next_enter = False  # type: ignore[attr-defined]
        await super()._on_key(event)

    def action_cursor_down(self, select: bool = False) -> None:  # type: ignore[override]
        if select:
            super().action_cursor_down(select)
            return
        app = self.app
        if getattr(getattr(app, "command_palette", None), "display", False):
            app.command_palette.move_next()  # type: ignore[union-attr]
            return
        super().action_cursor_down(select)

    def action_cursor_up(self, select: bool = False) -> None:  # type: ignore[override]
        if select:
            super().action_cursor_up(select)
            return
        app = self.app
        if getattr(getattr(app, "command_palette", None), "display", False):
            app.command_palette.move_previous()  # type: ignore[union-attr]
            return
        super().action_cursor_up(select)

    def action_cursor_end(self, select: bool = False) -> None:  # type: ignore[override]
        doc = getattr(self, "document", None)
        if doc is None:
            self.move_cursor((0, 0), select=select)
            return
        line_count = getattr(doc, "line_count", 0)
        if line_count <= 0:
            self.move_cursor((0, 0), select=select)
            return
        last_row = line_count - 1
        try:
            last_line = doc.get_line(last_row)
            last_col = len(last_line)
        except Exception:  # pragma: no cover - document API should not fail
            last_row, last_col = 0, 0
        self.move_cursor((last_row, last_col), select=select)
        self.scroll_cursor_visible()


class CommandHint(Static):
    def update_hint(self, paused: bool = False) -> None:
        suffix = ""
        if paused:
            suffix = " | [orange1]一時停止モード: ESCで巻き戻し、入力はワーカーへ送信[/]"
        self.update(
            "Commands : /attach, /parallel, /mode, /flow, /resume, /log, /status, /scoreboard, /done, /help, /exit | ESC: 一時停止/巻き戻し"
            + suffix
        )


class CommandPalette(Static):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.display = False
        self._items: List[PaletteItem] = []
        self._active_index: int = 0
        self._renderable: Text = Text()
        self._max_items: int = 7

    def set_items(self, items: List[PaletteItem]) -> None:
        self._items = items[: self._max_items]
        self._active_index = 0
        if not self._items:
            self.display = False
            self._renderable = Text()
            return
        self.display = True
        self._rebuild_renderable()

    def _rebuild_renderable(self) -> None:
        if not self._items:
            self._renderable = Text()
            return
        lines: List[Text] = []
        for idx, item in enumerate(self._items):
            prefix = "▶ " if idx == self._active_index else "  "
            style = "bold yellow" if idx == self._active_index else ""
            lines.append(Text(prefix + item.label, style=style))
        combined = Text()
        for idx, segment in enumerate(lines):
            if idx:
                combined.append("\n")
            combined.append(segment)
        self._renderable = combined
        self.refresh()

    def move_next(self) -> None:
        if not self._items:
            return
        self._active_index = (self._active_index + 1) % len(self._items)
        self._rebuild_renderable()

    def move_previous(self) -> None:
        if not self._items:
            return
        self._active_index = (self._active_index - 1) % len(self._items)
        self._rebuild_renderable()

    def get_active_item(self) -> Optional[PaletteItem]:
        if not self._items:
            return None
        return self._items[self._active_index]

    def render(self) -> Text:
        return self._renderable
