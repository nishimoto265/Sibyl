"""Textual-based interactive CLI for parallel developer orchestrator."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
import shutil
import subprocess
import time
from typing import Dict, List, Optional, Tuple

from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical
from textual.dom import NoScreen
from textual.message import Message
from textual.widgets import Footer, Header, OptionList
from textual.widgets.option_list import Option

from .controller import CLIController, SessionMode, build_orchestrator
from .controller.commands import CommandOption, CommandSuggestion
from .controller.events import ControllerEventType
from .stores import ManifestStore
from .ui.widgets import (
    CommandHint,
    CommandPalette,
    CommandTextArea,
    ControllerEvent,
    EventLog,
    PaletteItem,
    StatusPanel,
)


class ParallelDeveloperApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        layout: vertical;
        height: 1fr;
        padding: 1 2;
        min-height: 0;
    }

    #panel-stack {
        layout: vertical;
        height: 1fr;
        min-height: 0;
    }

    #log {
        height: 2fr;
        border: round $success;
        margin-bottom: 1;
        overflow-x: hidden;
        min-height: 3;
    }

    #log.paused {
        border: round $warning;
    }

    #status {
        border: round $success;
        padding: 1;
        height: 1fr;
        min-height: 1;
    }

    #status.paused {
        border: round $warning;
        color: $warning;
    }

    #selection {
        border: round $accent-darken-1;
        padding: 1;
        margin-bottom: 1;
    }

    #hint {
        padding: 1 0 0 0;
    }

    #command {
        margin-top: 1;
        height: auto;
        min-height: 3;
        overflow-x: hidden;
        border: round $success;
        background: $surface;
    }

    #command.paused {
        border: round $warning;
        background: $surface-lighten-3;
    }

    #hint.paused {
        color: $warning;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "終了"),
        ("escape", "close_palette", "一時停止/巻き戻し"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.status_panel: Optional[StatusPanel] = None
        self.log_panel: Optional[EventLog] = None
        self.selection_list: Optional[OptionList] = None
        self.command_input: Optional[CommandTextArea] = None
        self.command_palette: Optional[CommandPalette] = None
        self.command_hint: Optional[CommandHint] = None
        self._suppress_command_change: bool = False
        self._last_command_text: str = ""
        self._palette_mode: Optional[str] = None
        self._pending_command: Optional[str] = None
        self._default_placeholder: str = "指示または /コマンド"
        self._paused_placeholder: str = "一時停止中: ワーカーへの追加指示を入力"
        self._ctrl_c_armed: bool = False
        self._ctrl_c_armed_at: float = 0.0
        self._ctrl_c_timeout: float = 2.0
        self.controller = CLIController(
            event_handler=self._handle_controller_event,
            manifest_store=ManifestStore(),
            worktree_root=Path.cwd(),
            orchestrator_builder=build_orchestrator,
        )

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="body"):
            with Vertical(id="panel-stack"):
                self.status_panel = StatusPanel(id="status")
                yield self.status_panel
                self.log_panel = EventLog(id="log", max_lines=400)
                yield self.log_panel
            self.selection_list = OptionList(id="selection")
            self.selection_list._allow_focus = True
            self.selection_list.display = False
            yield self.selection_list
            self.command_palette = CommandPalette(id="command-palette")
            self.command_palette.display = False
            yield self.command_palette
            hint = CommandHint(id="hint")
            hint.update_hint(False)
            self.command_hint = hint
            yield hint
            self.command_input = CommandTextArea(
                text="",
                placeholder=self._default_placeholder,
                id="command",
                soft_wrap=True,
                tab_behavior="focus",
                show_line_numbers=False,
                highlight_cursor_line=False,
                compact=True,
            )
            yield self.command_input
        yield Footer()

    async def on_mount(self) -> None:
        if self.command_input:
            self.command_input.focus()
        self._post_event("status", {"message": "待機中"})
        self.set_class(False, "paused")
        if self.command_hint:
            self.command_hint.update_hint(False)

    def _submit_command_input(self) -> None:
        if not self.command_input:
            return
        value = self.command_input.text
        if self.command_palette and self.command_palette.display:
            item = self.command_palette.get_active_item()
            if item:
                asyncio.create_task(self._handle_palette_selection(item))
            return
        self._hide_command_palette()
        self._set_command_text("")
        self._ctrl_c_armed = False
        asyncio.create_task(self.controller.handle_input(value.rstrip("\n")))

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if not self.command_input or event.control is not self.command_input:
            return
        if self._suppress_command_change:
            return
        self.controller.history_reset()
        raw_value = self.command_input.text
        if raw_value == self._last_command_text:
            return
        self._last_command_text = raw_value
        processed = raw_value.replace("\n", "")
        if not processed:
            self._pending_command = None
            self._hide_command_palette()
            return
        if not processed.startswith("/"):
            self._pending_command = None
            self._hide_command_palette()
            return
        command, has_space, remainder = processed.partition(" ")
        command = command.lower()
        if not has_space:
            self._pending_command = None
            self._update_command_suggestions(command)
            return
        spec = self.controller._command_specs.get(command)
        if spec is None:
            self._hide_command_palette()
            return
        options = self.controller.get_command_options(command)
        if not options:
            self._last_command_text = value
            self._hide_command_palette()
            return
        remainder = remainder.strip()
        filtered: List[PaletteItem] = []
        for opt in options:
            label = self._format_option_label(opt)
            value_str = str(opt.value)
            search_text = (opt.label + " " + (opt.description or "")).lower()
            if (
                not remainder
                or value_str.startswith(remainder)
                or search_text.startswith(remainder.lower())
            ):
                filtered.append(PaletteItem(label, opt.value))
        if not filtered:
            self._hide_command_palette()
            return
        self._pending_command = command
        self._show_command_palette(filtered, mode="options")

    def _handle_controller_event(self, event_type: str, payload: Dict[str, object]) -> None:
        def _post() -> None:
            self.post_message(ControllerEvent(event_type, payload))

        try:
            self.call_from_thread(_post)
        except RuntimeError:
            _post()

    def _post_event(self, event_type: str, payload: Dict[str, object]) -> None:
        self.post_message(ControllerEvent(event_type, payload))

    def on_controller_event(self, event: ControllerEvent) -> None:
        event.stop()
        etype = event.event_type
        if etype == ControllerEventType.STATUS.value and self.status_panel:
            message = event.payload.get("message", "")
            self.status_panel.update_status(self.controller._config, str(message))
        elif etype == ControllerEventType.LOG.value and self.log_panel:
            text = str(event.payload.get("text", ""))
            self.log_panel.log(text)
        elif etype == ControllerEventType.SCOREBOARD.value:
            scoreboard = event.payload.get("scoreboard", {})
            if isinstance(scoreboard, dict):
                self._render_scoreboard(scoreboard)
        elif etype == ControllerEventType.LOG_COPY.value:
            message = self._copy_log_to_clipboard()
            self._notify_status(message)
        elif etype == ControllerEventType.LOG_SAVE.value:
            destination = str(event.payload.get("path", "") or "").strip()
            if not destination:
                self._notify_status("保存先パスが指定されていません。")
            else:
                message = self._save_log_to_path(destination)
                self._notify_status(message)
        elif etype == ControllerEventType.PAUSE_STATE.value:
            paused = bool(event.payload.get("paused", False))
            if self.status_panel:
                self.status_panel.set_class(paused, "paused")
            if self.log_panel:
                self.log_panel.set_class(paused, "paused")
            if self.command_input:
                self.command_input.set_class(paused, "paused")
                placeholder = self._paused_placeholder if paused else self._default_placeholder
                self.command_input.placeholder = placeholder
            if self.command_hint:
                self.command_hint.set_class(paused, "paused")
                self.command_hint.update_hint(paused)
        elif etype == ControllerEventType.SELECTION_REQUEST.value:
            candidates = event.payload.get("candidates", [])
            scoreboard = event.payload.get("scoreboard", {})
            self._render_scoreboard(scoreboard)
            if self.selection_list:
                self.selection_list.clear_options()
                for idx, candidate_label in enumerate(candidates, start=1):
                    option_text = self._build_option_label(candidate_label, scoreboard)
                    option = Option(option_text, str(idx))
                    self.selection_list.add_option(option)
                self.selection_list.display = True
                self.selection_list.focus()
                try:
                    self.selection_list.cursor_index = 0
                except AttributeError:
                    pass
            if self.command_input:
                self.command_input.display = False
        elif etype == ControllerEventType.SELECTION_FINISHED.value:
            if self.selection_list:
                self.selection_list.display = False
            if self.command_input:
                self.command_input.display = True
                self.command_input.focus()
        elif etype == ControllerEventType.QUIT.value:
            self.exit()

    async def action_quit(self) -> None:  # type: ignore[override]
        self.exit()

    def _handle_ctrl_c(self, event: events.Key) -> bool:
        key = (event.key or "").lower()
        name = (event.name or "").lower()
        if key not in {"ctrl+c", "control+c"} and name not in {"ctrl+c", "control+c"}:
            return False
        event.stop()
        now = time.monotonic()
        if self._ctrl_c_armed and now - self._ctrl_c_armed_at <= self._ctrl_c_timeout:
            self._ctrl_c_armed = False
            self.exit()
            return True

        self._ctrl_c_armed = True
        self._ctrl_c_armed_at = now
        if self.command_input:
            self._set_command_text("")
            cursor_reset = getattr(self.command_input, "action_cursor_line_start", None)
            if callable(cursor_reset):
                cursor_reset()
        self.controller.history_reset()
        return True

    def _render_scoreboard(self, scoreboard: Dict[str, Dict[str, object]]) -> None:
        if not self.log_panel:
            return
        if not scoreboard:
            self.log_panel.log("スコアボード情報はありません。")
            return
        lines = ["=== スコアボード ==="]
        for key, data in sorted(
            scoreboard.items(),
            key=lambda item: (item[1].get("score") is None, -(item[1].get("score") or 0.0)),
        ):
            score = data.get("score")
            comment = data.get("comment", "")
            selected = " [selected]" if data.get("selected") else ""
            score_text = "-" if score is None else f"{score:.2f}"
            lines.append(f"{key:>10}: {score_text}{selected} {comment}")
        self.log_panel.log("\n".join(lines))

    def _build_option_label(self, candidate_label: str, scoreboard: Dict[str, Dict[str, object]]) -> str:
        label_body = candidate_label.split(". ", 1)[1] if ". " in candidate_label else candidate_label
        key = label_body.split(" (", 1)[0].strip()
        entry = scoreboard.get(key, {})
        score = entry.get("score")
        comment = entry.get("comment", "")
        score_text = "-" if score is None else f"{score:.2f}"
        if comment:
            return f"{label_body} • {score_text} • {comment}"
        return f"{label_body} • {score_text}"

    def _set_command_text(self, value: str) -> None:
        if not self.command_input:
            return
        self._suppress_command_change = True
        self.command_input.text = value
        self._suppress_command_change = False
        self._last_command_text = value
        if not value:
            self._hide_command_palette()

    def _update_command_suggestions(self, prefix: str) -> None:
        suggestions = self.controller.get_command_suggestions(prefix)
        if not suggestions:
            self._hide_command_palette()
            return
        items = [PaletteItem(f"{s.name:<10} {s.description}", s.name) for s in suggestions]
        self._show_command_palette(items, mode="command")

    def _format_option_label(self, option: CommandOption) -> str:
        display = getattr(option, "display", None)
        if display:
            return display
        description = getattr(option, "description", None)
        if description:
            return f"{option.label} - {description}"
        return option.label

    def _show_command_palette(self, items: List[PaletteItem], *, mode: str) -> None:
        if not self.command_palette:
            return
        if not items:
            self._hide_command_palette()
            return
        self._palette_mode = mode
        self.command_palette.set_items(items)
        if self.command_input and self.command_input.has_focus:
            self.set_focus(self.command_input)

    def _hide_command_palette(self) -> None:
        if self.command_palette:
            self.command_palette.display = False
            self.command_palette.set_items([])
        self._palette_mode = None
        self._pending_command = None
        if self.command_input:
            self.command_input.focus()

    def action_close_palette(self) -> None:
        self._hide_command_palette()
        self.controller.handle_escape()

    def action_palette_next(self) -> None:
        if self.command_palette and self.command_palette.display:
            self.command_palette.move_next()

    def action_palette_previous(self) -> None:
        if self.command_palette and self.command_palette.display:
            self.command_palette.move_previous()

    def _collect_log_text(self) -> Tuple[str, bool]:
        if not self.log_panel:
            return "", False
        selection = None
        with suppress(NoScreen):
            selection = self.log_panel.text_selection
        if selection:
            extracted = self.log_panel.get_selection(selection)
            if extracted:
                text, ending = extracted
                final_text = text if ending is None else f"{text}{ending}"
                return final_text.rstrip("\n"), True
        if isinstance(self.log_panel, EventLog):
            lines = self.log_panel.entries
        else:
            lines = list(getattr(self.log_panel, "lines", []))
            if lines and lines[-1] == "":
                lines = lines[:-1]
        text = "\n".join(line.rstrip() for line in lines).rstrip("\n")
        return text, False

    def _copy_log_to_clipboard(self) -> str:
        text, from_selection = self._collect_log_text()
        if not text:
            return "コピー対象のログがありません。"
        self.copy_to_clipboard(text)
        self._copy_to_system_clipboard(text)
        if from_selection:
            return "選択範囲をクリップボードへコピーしました。"
        return "ログ全体をクリップボードへコピーしました。"

    def _copy_to_system_clipboard(self, text: str) -> None:
        commands = []
        if shutil.which("pbcopy"):
            commands.append(["pbcopy"])
        if shutil.which("wl-copy"):
            commands.append(["wl-copy"])
        if shutil.which("xclip"):
            commands.append(["xclip", "-selection", "clipboard"])
        if shutil.which("clip.exe"):
            commands.append(["clip.exe"])

        for command in commands:
            try:
                subprocess.run(
                    command,
                    input=text.encode("utf-8"),
                    check=True,
                )
                break
            except Exception:
                continue

    def _save_log_to_path(self, destination: str) -> str:
        text, _ = self._collect_log_text()
        if not text:
            return "保存対象のログがありません。"
        try:
            path = Path(destination).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            return f"ログの保存に失敗しました: {exc}"
        return f"ログを {path} に保存しました。"

    def _notify_status(self, message: str, *, also_log: bool = True) -> None:
        if self.status_panel:
            self.status_panel.update_status(self.controller._config, message)
        if also_log and self.log_panel:
            self.log_panel.log(message)

    def _handle_tab_navigation(self, *, reverse: bool = False) -> None:
        palette = self.command_palette
        palette_visible = bool(palette and palette.display)
        if palette_visible:
            if reverse:
                self.action_palette_previous()
            else:
                self.action_palette_next()
        if self.command_input:
            self.command_input.focus()

    def on_key(self, event: events.Key) -> None:
        if self._handle_ctrl_c(event):
            return
        if self._handle_text_shortcuts(event):
            return
        key_value = (event.key or "").lower()
        name_value = (event.name or "").lower()
        if key_value in {"tab", "shift+tab"} or name_value in {"tab", "shift_tab"}:
            event.stop()
            event.prevent_default()
            self._handle_tab_navigation(reverse=key_value == "shift+tab" or name_value == "shift_tab")
            return

    def _handle_text_shortcuts(self, event: events.Key) -> bool:
        shortcuts_select_all = {
            "ctrl+a",
            "control+a",
            "cmd+a",
            "command+a",
            "meta+a",
            "ctrl+shift+a",
            "control+shift+a",
        }
        shortcuts_copy = {
            "ctrl+c",
            "control+c",
            "cmd+c",
            "command+c",
            "meta+c",
            "ctrl+shift+c",
            "control+shift+c",
            "cmd+shift+c",
            "command+shift+c",
            "meta+shift+c",
            "ctrl+alt+c",
            "control+alt+c",
            "cmd+alt+c",
            "command+alt+c",
            "meta+alt+c",
        }

        def matches(shortcuts: set[str]) -> bool:
            key_value = event.key.lower()
            name_value = (event.name or "").lower()
            if key_value in shortcuts:
                return True
            if name_value and name_value in {shortcut.replace("+", "_") for shortcut in shortcuts}:
                return True
            return False

        if event.key in {"up", "down"} and not (self.command_palette and self.command_palette.display):
            history_text = self.controller.history_previous() if event.key == "up" else self.controller.history_next()
            if history_text is not None:
                if self.command_input:
                    self._set_command_text(history_text)
                    self.command_input.action_cursor_end()
                event.stop()
                return True

        if matches(shortcuts_select_all):
            if self.log_panel:
                self.log_panel.text_select_all()
                self._notify_status("ログ全体を選択しました。", also_log=False)
            event.stop()
            return True
        if matches(shortcuts_copy) and self.log_panel:
            message = self._copy_log_to_clipboard()
            self._notify_status(message)
            event.stop()
            return True
        return False

    def on_click(self, event: events.Click) -> None:
        if not self.command_input:
            return
        control = event.control
        if control is None:
            self.set_focus(self.command_input)
            return

        def within(widget: Optional[Widget]) -> bool:
            return bool(widget and widget in control.ancestors_with_self)

        if within(self.command_input):
            return
        if self.log_panel and within(self.log_panel):
            return
        if self.selection_list and self.selection_list.display and within(self.selection_list):
            return
        if self.command_palette and self.command_palette.display and within(self.command_palette):
            return
        self.set_focus(self.command_input)

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self.selection_list and event.option_list is self.selection_list:
            event.stop()
            try:
                index = int(event.option_id)
            except (TypeError, ValueError):
                return
            self.controller._resolve_selection(index)
            return
        if self.command_palette and self.command_palette.display:
            event.stop()
            item = self.command_palette.get_active_item()
            if item:
                await self._handle_palette_selection(item)

    async def _handle_palette_selection(self, item: PaletteItem) -> None:
        if self._palette_mode == "command":
            command_name = str(item.value)
            options = self.controller.get_command_options(command_name)
            if options:
                self._pending_command = command_name
                option_items = [PaletteItem(opt.label, opt.value) for opt in options]
                self._show_command_palette(option_items, mode="options")
                if self.command_input:
                    self._set_command_text(f"{command_name} ")
                return
            if self.command_input:
                self._set_command_text("")
            self._hide_command_palette()
            await self.controller.execute_command(command_name)
            return
        if self._palette_mode == "options" and self._pending_command:
            command_name = self._pending_command
            value = item.value
            self._pending_command = None
            if self.command_input:
                self._set_command_text("")
            self._hide_command_palette()
            await self.controller.execute_command(command_name, value)


def run() -> None:
    ParallelDeveloperApp().run()
