"""Textual-based interactive CLI for parallel developer orchestrator."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import Future
import importlib.util
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Mapping
import platform
import subprocess
import shlex
import shutil
from subprocess import PIPE

from textual import events
from textual.app import App, ComposeResult
from rich.text import Text
from contextlib import suppress

from textual.containers import Container, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Footer, Header, OptionList, RichLog, Static, TextArea
from textual.widgets.option_list import Option
from textual.dom import NoScreen

from .orchestrator import CandidateInfo, CycleLayout, OrchestrationResult, Orchestrator, SelectionDecision
from .session_manifest import ManifestStore, PaneRecord, SessionManifest, SessionReference
from .services import CodexMonitor, LogManager, TmuxLayoutManager, WorktreeManager

_UI_MODULE_PATH = Path(__file__).with_name("25-11-03-05_ui.py")
_ui_spec = importlib.util.spec_from_file_location("parallel_developer.ui_widgets", _UI_MODULE_PATH)
if _ui_spec is None or _ui_spec.loader is None:
    raise RuntimeError(f"UI module not found at {_UI_MODULE_PATH}")
_ui_module = importlib.util.module_from_spec(_ui_spec)
sys.modules[_ui_spec.name] = _ui_module  # type: ignore[index]
_ui_spec.loader.exec_module(_ui_module)
CommandHint = _ui_module.CommandHint  # type: ignore[attr-defined]
CommandPalette = _ui_module.CommandPalette  # type: ignore[attr-defined]
CommandTextArea = _ui_module.CommandTextArea  # type: ignore[attr-defined]
ControllerEvent = _ui_module.ControllerEvent  # type: ignore[attr-defined]
EventLog = _ui_module.EventLog  # type: ignore[attr-defined]
PaletteItem = _ui_module.PaletteItem  # type: ignore[attr-defined]
StatusPanel = _ui_module.StatusPanel  # type: ignore[attr-defined]


class SessionMode(str, Enum):
    PARALLEL = "parallel"
    MAIN = "main"


class TmuxAttachManager:
    """Launch an external terminal to attach to the tmux session."""

    def attach(self, session_name: str, workdir: Optional[Path] = None) -> subprocess.CompletedProcess:
        system = platform.system().lower()
        command_string = self._build_command_string(session_name, workdir)

        if "darwin" in system:
            escaped_command = self._escape_for_applescript(command_string)
            apple_script = (
                'tell application "Terminal"\n'
                f'    do script "{escaped_command}"\n'
                "    activate\n"
                "end tell"
            )
            command = ["osascript", "-e", apple_script]
            try:
                return subprocess.run(command, check=False)
            except FileNotFoundError:
                # Fall through to generic fallback below.
                pass
        elif "linux" in system:
            command = ["gnome-terminal", "--", "bash", "-lc", command_string]
            try:
                return subprocess.run(command, check=False)
            except FileNotFoundError:
                # gnome-terminal not available; fall back to shell attach.
                pass

        fallback_command: List[str]
        if shutil.which("bash"):
            fallback_command = ["bash", "-lc", command_string]
        else:
            fallback_command = ["tmux", "attach", "-t", session_name]

        try:
            return subprocess.run(fallback_command, check=False)
        except FileNotFoundError:
            return subprocess.CompletedProcess(fallback_command, returncode=127)

    def is_attached(self, session_name: str) -> bool:
        try:
            result = subprocess.run(
                [
                    "tmux",
                    "display-message",
                    "-t",
                    session_name,
                    "-p",
                    "#{session_attached}",
                ],
                check=False,
                stdout=PIPE,
                stderr=PIPE,
                text=True,
            )
        except FileNotFoundError:
            return False

        if result.returncode != 0:
            return False
        output = (result.stdout or "").strip().lower()
        return output in {"1", "true"}

    def session_exists(self, session_name: str) -> bool:
        try:
            result = subprocess.run(
                [
                    "tmux",
                    "has-session",
                    "-t",
                    session_name,
                ],
                check=False,
                stdout=PIPE,
                stderr=PIPE,
                text=True,
            )
        except FileNotFoundError:
            return False
        return result.returncode == 0

    def _build_command_string(self, session_name: str, workdir: Optional[Path]) -> str:
        return f"tmux attach -t {shlex.quote(session_name)}"

    @staticmethod
    def _escape_for_applescript(command: str) -> str:
        return command.replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class SessionConfig:
    session_id: str
    tmux_session: str
    worker_count: int
    mode: SessionMode
    logs_root: Path
    reuse_existing_session: bool = False

@dataclass
class SelectionContext:
    future: Future
    candidates: List[CandidateInfo]
    scoreboard: Dict[str, Dict[str, object]]


@dataclass
class CommandSuggestion:
    name: str
    description: str


@dataclass
class CommandOption:
    label: str
    value: object


class CLIController:
    """Core orchestration controller decoupled from Textual UI."""

    def __init__(
        self,
        *,
        event_handler: Callable[[str, Dict[str, object]], None],
        orchestrator_builder: Callable[..., Orchestrator] = None,
        manifest_store: Optional[ManifestStore] = None,
        worktree_root: Optional[Path] = None,
        settings_path: Optional[Path] = None,
    ) -> None:
        self._event_handler = event_handler
        self._builder = orchestrator_builder or self._default_builder
        self._manifest_store = manifest_store or ManifestStore()
        self._worktree_root = Path(worktree_root or Path.cwd())
        self._config = self._create_initial_config()
        self._last_scoreboard: Dict[str, Dict[str, object]] = {}
        self._last_instruction: Optional[str] = None
        self._running: bool = False
        self._selection_context: Optional[SelectionContext] = None
        self._resume_options: List[SessionReference] = []
        self._last_selected_session: Optional[str] = None
        self._active_main_session_id: Optional[str] = None
        self._paused: bool = False
        self._cycle_history: List[Dict[str, object]] = []
        self._input_history: List[str] = []
        self._history_cursor: int = 0
        self._cycle_counter: int = 0
        self._current_cycle_id: Optional[int] = None
        self._cancelled_cycles: Set[int] = set()
        self._last_tmux_manager: Optional[TmuxLayoutManager] = None
        self._active_orchestrator: Optional[Orchestrator] = None
        self._queued_instruction: Optional[str] = None
        self._continue_future: Optional[Future] = None
        self._attach_manager = TmuxAttachManager()
        self._settings_path = Path(settings_path) if settings_path else (self._worktree_root / ".parallel-dev" / "settings.json")
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings: Dict[str, object] = self._load_settings()
        self._attach_mode: str = str(self._settings.get("attach_mode", "auto"))
        self._command_specs: Dict[str, Dict[str, object]] = {
            "/attach": {
                "description": "tmux セッションへの接続モードを切り替える、または即座に接続する",
                "options": [
                    CommandOption("auto", "auto"),
                    CommandOption("manual", "manual"),
                    CommandOption("now", "now"),
                ],
            },
            "/parallel": {
                "description": "ワーカー数を設定する",
                "options": [CommandOption(str(n), str(n)) for n in range(1, 5)],
            },
            "/mode": {
                "description": "実行モードを切り替える",
                "options": [
                    CommandOption("main", "main"),
                    CommandOption("parallel", "parallel"),
                ],
            },
            "/resume": {
                "description": "保存済みセッションを再開する",
                "options_provider": self._build_resume_options,
            },
            "/continue": {
                "description": "現行サイクルでワーカーへの追加指示を続ける",
            },
            "/log": {
                "description": "ログをコピーするかファイルへ保存する",
                "options": [
                    CommandOption("copy", "copy"),
                    CommandOption("save", "save"),
                ],
            },
            "/status": {"description": "現在の状態を表示する"},
            "/scoreboard": {"description": "直近のスコアボードを表示する"},
            "/done": {"description": "全ワーカーに /done を送信して採点フェーズへ移行する"},
            "/help": {"description": "コマンド一覧を表示する"},
            "/exit": {"description": "CLI を終了する"},
        }

    async def handle_input(self, user_input: str) -> None:
        text = user_input.strip()
        if not text:
            return
        if text.startswith("/"):
            self._record_history(text)
            await self._execute_text_command(text)
            return
        if self._paused:
            self._record_history(text)
            await self._dispatch_paused_instruction(text)
            return
        if self._running:
            if self._current_cycle_id and self._current_cycle_id in self._cancelled_cycles:
                self._queued_instruction = text
                self._emit("log", {"text": "キャンセル処理中です。完了後にこの指示を実行します。"})
                return
        if self._running:
            self._emit("log", {"text": "別の指示を処理中です。完了を待ってから再度実行してください。"})
            return
        self._record_history(text)
        await self._run_instruction(text)

    async def _execute_text_command(self, command_text: str) -> None:
        parts = command_text.split(maxsplit=1)
        if not parts:
            return
        name = parts[0].lower()
        if name == "/quit":
            name = "/exit"
        option = parts[1].strip() if len(parts) > 1 else None
        if option == "":
            option = None
        await self.execute_command(name, option)

    def get_command_suggestions(self, prefix: str) -> List[CommandSuggestion]:
        prefix = (prefix or "/").lower()
        if not prefix.startswith("/"):
            prefix = "/" + prefix
        suggestions: List[CommandSuggestion] = []
        for name in sorted(self._command_specs.keys()):
            if name.startswith(prefix):
                spec = self._command_specs[name]
                suggestions.append(CommandSuggestion(name=name, description=spec["description"]))
        if not suggestions and prefix == "/":
            for name in sorted(self._command_specs.keys()):
                spec = self._command_specs[name]
                suggestions.append(CommandSuggestion(name=name, description=spec["description"]))
        return suggestions

    def get_command_options(self, name: str) -> List[CommandOption]:
        spec = self._command_specs.get(name)
        if not spec:
            return []
        if "options_provider" in spec:
            return spec["options_provider"]()
        if "options" in spec:
            return [CommandOption(item.label, item.value) for item in spec["options"]]
        return []

    async def execute_command(self, name: str, option: Optional[object] = None) -> None:
        spec = self._command_specs.get(name)
        if spec is None:
            self._emit("log", {"text": f"未知のコマンドです: {name}"})
            return

        if name == "/exit":
            self._emit("quit", {})
            return

        if name == "/help":
            help_lines = ["利用可能なコマンド:"]
            for cmd in sorted(self._command_specs.keys()):
                help_lines.append(f"  {cmd:10s} : {self._command_specs[cmd]['description']}")
            self._emit("log", {"text": "\n".join(help_lines)})
            return

        if name == "/status":
            self._emit_status("待機中")
            return

        if name == "/scoreboard":
            self._emit("scoreboard", {"scoreboard": self._last_scoreboard})
            return

        if name == "/done":
            if self._continue_future and not self._continue_future.done():
                self._continue_future.set_result(False)
                self._emit("log", {"text": "/done を検知として扱い、採点フェーズへ進みます。"})
                return
            if self._active_orchestrator:
                count = self._active_orchestrator.force_complete_workers()
                if count:
                    self._emit("log", {"text": f"/done を検知として扱い、{count} ワーカーを完了済みに設定しました。"})
                else:
                    self._emit("log", {"text": "完了扱いにするワーカーセッションが見つかりませんでした。"})
            else:
                self._emit("log", {"text": "現在進行中のワーカークセッションがないため /done を適用できません。"})
            return

        if name == "/continue":
            if self._continue_future and not self._continue_future.done():
                self._continue_future.set_result(True)
                self._emit("log", {"text": "/continue を受け付けました。ワーカーに追加指示を送れます。"})
            else:
                self._emit("log", {"text": "/continue は現在利用できません。"})
            return

        if name == "/attach":
            mode = (str(option).lower() if option is not None else None)
            if mode in {"auto", "manual"}:
                self._attach_mode = mode
                self._emit("log", {"text": f"/attach モードを {mode} に設定しました。"})
                self._save_settings()
                return
            if mode == "now" or option is None:
                await self._handle_attach_command(force=True)
                return
            self._emit("log", {"text": "使い方: /attach [auto|manual|now]"})
            return

        if name == "/parallel":
            if option is None:
                self._emit("log", {"text": "使い方: /parallel <ワーカー数>"})
                return
            try:
                value = int(str(option))
            except ValueError:
                self._emit("log", {"text": "ワーカー数は数字で指定してください。"})
                return
            if value < 1:
                self._emit("log", {"text": "ワーカー数は1以上で指定してください。"})
                return
            self._config.worker_count = value
            self._emit_status("設定を更新しました。")
            return

        if name == "/mode":
            mode = (str(option).lower() if option is not None else None)
            if mode not in {"main", "parallel"}:
                self._emit("log", {"text": "使い方: /mode main | /mode parallel"})
                return
            self._config.mode = SessionMode(mode)
            self._emit_status("設定を更新しました。")
            return

        if name == "/resume":
            if option is None:
                self._list_sessions()
                return
            index: Optional[int] = None
            if isinstance(option, int):
                index = option
            else:
                try:
                    index = int(str(option))
                except ValueError:
                    index = self._find_resume_index_by_session(str(option))
            if index is None:
                self._emit("log", {"text": "指定されたセッションが見つかりません。"})
                return
            self._load_session(index)
            return

        if name == "/log":
            if option is None:
                self._emit(
                    "log",
                    {
                        "text": "使い方: /log copy | /log save <path>\n"
                        "  copy : 現在のログをクリップボードへコピー\n"
                        "  save : 指定パスへログを書き出す"
                    },
                )
                return
            action: str
            argument: Optional[str] = None
            if isinstance(option, str):
                sub_parts = option.split(maxsplit=1)
                action = sub_parts[0].lower()
                if len(sub_parts) > 1:
                    argument = sub_parts[1].strip()
            else:
                action = str(option).lower()
            if action == "copy":
                self._emit("log_copy", {})
                return
            if action == "save":
                if not argument:
                    self._emit("log", {"text": "保存先パスを指定してください。例: /log save logs/output.log"})
                    return
                self._emit("log_save", {"path": argument})
                return
            self._emit("log", {"text": "使い方: /log copy | /log save <path>"})
            return

    def _find_resume_index_by_session(self, token: str) -> Optional[int]:
        if not self._resume_options:
            self._resume_options = self._manifest_store.list_sessions()
        for idx, ref in enumerate(self._resume_options, start=1):
            if ref.session_id == token or ref.session_id.startswith(token):
                return idx
        return None

    def broadcast_escape(self) -> None:
        session_name = self._config.tmux_session
        pane_ids = self._tmux_list_panes()
        if pane_ids is None:
            return
        if not pane_ids:
            self._emit("log", {"text": f"tmuxセッション {session_name} にペインが見つかりませんでした。"})
            return

        for pane_id in pane_ids:
            subprocess.run(
                ["tmux", "send-keys", "-t", pane_id, "Escape"],
                check=False,
            )
        self._emit("log", {"text": f"tmuxセッション {session_name} の {len(pane_ids)} 個のペインへEscapeを送信しました。"})

    def handle_escape(self) -> None:
        self.broadcast_escape()
        if not self._paused:
            self._paused = True
            self._emit("log", {"text": "一時停止モードに入りました。追加指示は現在のワーカーペインへ送信されます。"})
            self._emit_status("一時停止モード")
            self._emit_pause_state()
            return
        if self._running:
            current_id = self._current_cycle_id
            if current_id is not None:
                self._cancelled_cycles.add(current_id)
            self._current_cycle_id = None
            self._running = False
        if self._continue_future and not self._continue_future.done():
            self._continue_future.set_result(False)
        self._paused = False
        self._emit("log", {"text": "現在のサイクルをキャンセルし、前の状態へ戻しました。"})
        self._emit_status("待機中")
        self._emit_pause_state()
        self._perform_revert(silent=True)
        return
    def _tmux_list_panes(self) -> Optional[List[str]]:
        session_name = self._config.tmux_session
        try:
            result = subprocess.run(
                ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_id}"],
                check=False,
                stdout=PIPE,
                stderr=PIPE,
                text=True,
            )
        except FileNotFoundError:
            self._emit("log", {"text": "tmux コマンドが見つかりません。tmuxがインストールされているか確認してください。"})
            return None
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            if message:
                self._emit("log", {"text": f"tmux list-panes に失敗しました: {message}"})
            return None
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    async def _dispatch_paused_instruction(self, instruction: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._send_instruction_to_panes(instruction))

    def _send_instruction_to_panes(self, instruction: str) -> None:
        session_name = self._config.tmux_session
        pane_ids = self._tmux_list_panes()
        if pane_ids is None:
            return
        if len(pane_ids) <= 2:
            self._emit("log", {"text": f"tmuxセッション {session_name} にワーカーペインが見つからず、追加指示を送信できませんでした。"})
            return
        worker_panes = pane_ids[2:]
        for pane_id in worker_panes:
            subprocess.run(
                ["tmux", "send-keys", "-t", pane_id, instruction, "Enter"],
                check=False,
            )
        preview = instruction.replace("\n", " ")[:60]
        if len(instruction) > 60:
            preview += "..."
        self._emit("log", {"text": f"[pause] {len(worker_panes)} ワーカーペインへ追加指示を送信: {preview}"})
        self._paused = False
        self._emit_pause_state()
        self._emit_status("待機中")

    def _record_cycle_snapshot(self, result: OrchestrationResult, cycle_id: int) -> None:
        snapshot = {
            "cycle_id": cycle_id,
            "selected_session": result.selected_session,
            "scoreboard": dict(result.sessions_summary),
            "instruction": self._last_instruction,
        }
        self._cycle_history.append(snapshot)

    def _handle_worker_decision(
        self,
        fork_map: Mapping[str, str],
        completion_info: Mapping[str, Any],
        layout: CycleLayout,
    ) -> bool:
        future = Future()
        self._continue_future = future
        self._emit(
            "log",
            {
                "text": (
                    "ワーカーの処理が完了しました。追加で作業させるには /continue を、"
                    "評価へ進むには /done を入力してください。"
                )
            },
        )
        try:
            decision = future.result()
        finally:
            self._continue_future = None
        return bool(decision)

    def _record_history(self, text: str) -> None:
        entry = text.strip()
        if not entry:
            return
        if self._input_history and self._input_history[-1] == entry:
            self._history_cursor = len(self._input_history)
            return
        self._input_history.append(entry)
        self._history_cursor = len(self._input_history)

    def history_previous(self) -> Optional[str]:
        if not self._input_history:
            return None
        if self._history_cursor > 0:
            self._history_cursor -= 1
        return self._input_history[self._history_cursor]

    def history_next(self) -> Optional[str]:
        if not self._input_history:
            return None
        if self._history_cursor < len(self._input_history) - 1:
            self._history_cursor += 1
            return self._input_history[self._history_cursor]
        self._history_cursor = len(self._input_history)
        return ""

    def history_reset(self) -> None:
        self._history_cursor = len(self._input_history)

    def _perform_revert(self, silent: bool = False) -> None:
        tmux_manager = self._last_tmux_manager
        pane_ids = self._tmux_list_panes() or []
        main_pane = pane_ids[0] if pane_ids else None

        if not self._cycle_history:
            session_id = self._active_main_session_id
            self._last_selected_session = session_id
            self._active_main_session_id = session_id
            self._last_scoreboard = {}
            self._last_instruction = None
            self._paused = False
            if tmux_manager and main_pane:
                if session_id:
                    tmux_manager.promote_to_main(session_id=session_id, pane_id=main_pane)
                else:
                    tmux_manager.launch_main_session(pane_id=main_pane)
            summary = session_id or "(未選択)"
            if not silent:
                self._emit("log", {"text": f"前回のセッションを再開しました。次の指示はセッション {summary} から再開します。"})
                self._emit_status("待機中")
                self._emit_pause_state()
            return

        self._cycle_history.pop()
        snapshot = self._cycle_history[-1] if self._cycle_history else None
        session_id = snapshot.get("selected_session") if snapshot else self._active_main_session_id

        self._last_selected_session = session_id
        self._active_main_session_id = session_id
        self._last_scoreboard = snapshot.get("scoreboard", {}) if snapshot else {}
        self._last_instruction = snapshot.get("instruction") if snapshot else None
        if self._last_scoreboard:
            self._emit("scoreboard", {"scoreboard": self._last_scoreboard})

        self._paused = False
        if tmux_manager and main_pane:
            if session_id:
                tmux_manager.promote_to_main(session_id=session_id, pane_id=main_pane)
            else:
                tmux_manager.launch_main_session(pane_id=main_pane)

        summary = session_id or "(未選択)"
        if not silent:
            self._emit("log", {"text": f"サイクルを巻き戻しました。次の指示はセッション {summary} から再開します。"})
            self._emit_status("待機中")
            self._emit_pause_state()

    def _emit_pause_state(self) -> None:
        self._emit("pause_state", {"paused": self._paused})

    def _on_main_session_started(self, session_id: str) -> None:
        self._active_main_session_id = session_id
        if self._last_selected_session is None:
            self._last_selected_session = session_id
        self._config.reuse_existing_session = True

    async def _run_instruction(self, instruction: str) -> None:
        if self._running:
            self._emit("log", {"text": "別の指示を処理中です。完了を待ってから再度実行してください。"})
            return
        if self._selection_context:
            self._emit("log", {"text": "候補選択待ちです。/pick <n> で選択してください。"})
            return

        self._cycle_counter += 1
        cycle_id = self._cycle_counter
        self._current_cycle_id = cycle_id
        self._running = True
        self._emit_status("メインセッションを準備中...")
        self._active_main_session_id = None

        logs_dir = self._create_cycle_logs_dir()

        orchestrator = self._builder(
            worker_count=self._config.worker_count,
            log_dir=logs_dir,
            session_name=self._config.tmux_session,
            reuse_existing_session=self._config.reuse_existing_session,
        )
        self._active_orchestrator = orchestrator
        self._last_tmux_manager = getattr(orchestrator, "_tmux", None)
        main_hook = getattr(orchestrator, "set_main_session_hook", None)
        if callable(main_hook):
            main_hook(self._on_main_session_started)
        worker_decider = getattr(orchestrator, "set_worker_decider", None)
        if callable(worker_decider):
            worker_decider(self._handle_worker_decision)

        loop = asyncio.get_running_loop()

        def selector(candidates: List[CandidateInfo], scoreboard: Optional[Dict[str, Dict[str, object]]] = None) -> SelectionDecision:
            future: Future = Future()
            context = SelectionContext(
                future=future,
                candidates=candidates,
                scoreboard=scoreboard or {},
            )
            self._selection_context = context
            formatted = [f"{idx + 1}. {candidate.label}" for idx, candidate in enumerate(candidates)]
            self._emit(
                "selection_request",
                {
                    "candidates": formatted,
                    "scoreboard": scoreboard or {},
                },
            )
            return future.result()

        resume_session = self._last_selected_session

        def run_cycle() -> OrchestrationResult:
            return orchestrator.run_cycle(
                instruction,
                selector=selector,
                resume_session_id=resume_session,
            )

        auto_attach_task: Optional[asyncio.Task[None]] = None
        cancelled = False
        continued = False
        try:
            self._emit("log", {"text": f"指示を開始: {instruction}"})
            if self._attach_mode == "auto":
                auto_attach_task = asyncio.create_task(self._handle_attach_command(force=False))
            result: OrchestrationResult = await loop.run_in_executor(None, run_cycle)
            continued = getattr(result, "continue_requested", False)
            if continued:
                self._last_selected_session = result.selected_session
                self._active_main_session_id = result.selected_session
                self._config.reuse_existing_session = True
                self._last_scoreboard = {}
                self._emit("log", {"text": "ワーカーを継続します。新しい指示を入力してください。"})
            elif cycle_id in self._cancelled_cycles:
                cancelled = True
                self._cancelled_cycles.discard(cycle_id)
            else:
                self._last_scoreboard = dict(result.sessions_summary)
                self._last_instruction = instruction
                self._last_selected_session = result.selected_session
                self._active_main_session_id = result.selected_session
                self._config.reuse_existing_session = True
                self._emit("scoreboard", {"scoreboard": self._last_scoreboard})
                self._emit("log", {"text": "指示が完了しました。"})
                if result.artifact:
                    manifest = self._build_manifest(result, logs_dir)
                    self._manifest_store.save_manifest(manifest)
                    self._emit("log", {"text": f"セッションを保存しました: {manifest.session_id}"})
                self._record_cycle_snapshot(result, cycle_id)
        except Exception as exc:  # pylint: disable=broad-except
            self._emit("log", {"text": f"エラーが発生しました: {exc}"})
        finally:
            self._selection_context = None
            if self._current_cycle_id == cycle_id:
                self._current_cycle_id = None
            self._running = False
            if cancelled:
                self._emit_status("待機中")
                self._emit_pause_state()
                self._perform_revert(silent=True)
            else:
                self._emit_status("一時停止中" if self._paused else "待機中")
                self._emit_pause_state()
            if auto_attach_task:
                try:
                    await auto_attach_task
                except Exception:  # pragma: no cover - logging handled inside
                    self._emit("log", {"text": "[auto] tmuxへの接続処理でエラーが発生しました。"})
            if self._active_orchestrator is orchestrator:
                self._active_orchestrator = None
            if cancelled:
                queued = self._queued_instruction
                self._queued_instruction = None
                if queued:
                    asyncio.create_task(self.handle_input(queued))
                return
            if continued:
                return

    def _resolve_selection(self, index: int) -> None:
        if not self._selection_context:
            self._emit("log", {"text": "現在選択待ちではありません。"})
            return
        context = self._selection_context
        if index < 1 or index > len(context.candidates):
            self._emit("log", {"text": "無効な番号です。"})
            return
        candidate = context.candidates[index - 1]
        scores = {
            cand.key: (1.0 if cand.key == candidate.key else 0.0) for cand in context.candidates
        }
        decision = SelectionDecision(selected_key=candidate.key, scores=scores)
        context.future.set_result(decision)
        self._emit("log", {"text": f"{candidate.label} を選択しました。"})
        self._selection_context = None
        self._emit("selection_finished", {})

    async def _handle_attach_command(self, *, force: bool = False) -> None:
        session_name = self._config.tmux_session
        wait_for_session = not force and self._attach_mode == "auto"
        if wait_for_session:
            self._emit("log", {"text": f"[auto] tmuxセッション {session_name} の起動を待機中..."})
            session_ready = await self._wait_for_session(session_name)
            if not session_ready:
                self._emit(
                    "log",
                    {"text": f"[auto] tmuxセッション {session_name} が見つかりませんでした。少し待ってから再度試してください。"},
                )
                return
        else:
            if not self._attach_manager.session_exists(session_name):
                self._emit(
                    "log",
                    {
                        "text": (
                            f"tmuxセッション {session_name} がまだ存在しません。"
                            " 指示を送信してセッションを初期化した後に再度実行してください。"
                        )
                    },
                )
                return

        perform_detection = not force and self._attach_mode == "auto"
        if perform_detection:
            if self._attach_manager.is_attached(session_name):
                self._emit(
                    "log",
                    {"text": f"[auto] tmuxセッション {session_name} は既に接続済みのため、自動アタッチをスキップしました。"},
                )
                return
        result = self._attach_manager.attach(session_name, workdir=self._worktree_root)
        if result.returncode == 0:
            prefix = "[auto] " if perform_detection else ""
            self._emit("log", {"text": f"{prefix}tmuxセッション {session_name} に接続しました。"})
        else:
            self._emit("log", {"text": "tmuxへの接続に失敗しました。tmuxが利用可能か確認してください。"})

    def _build_resume_options(self) -> List[CommandOption]:
        references = self._manifest_store.list_sessions()
        self._resume_options = references
        options: List[CommandOption] = []
        for idx, ref in enumerate(references, start=1):
            summary = ref.latest_instruction or ""
            label = f"{idx}. {ref.created_at} | tmux={ref.tmux_session}"
            if summary:
                label += f" | last: {summary[:40]}"
            options.append(CommandOption(label, idx))
        return options

    def _list_sessions(self) -> None:
        references = self._manifest_store.list_sessions()
        self._resume_options = references
        if not references:
            self._emit("log", {"text": "保存済みセッションが見つかりません。"})
            return
        lines = [
            "=== 保存されたセッション ===",
        ]
        for idx, ref in enumerate(references, start=1):
            summary = ref.latest_instruction or ""
            lines.append(
                f"{idx}. {ref.created_at} | tmux={ref.tmux_session} | workers={ref.worker_count} | mode={ref.mode}"
            )
            if summary:
                lines.append(f"   last instruction: {summary[:80]}")
        self._emit("log", {"text": "\n".join(lines)})
        self._emit("log", {"text": "再開するには /resume からセッションを選択してください。"})

    def _load_session(self, index: int) -> None:
        if not self._resume_options:
            self._emit("log", {"text": "先に /resume で一覧を表示してください。"})
            return
        if index < 1 or index > len(self._resume_options):
            self._emit("log", {"text": "無効な番号です。"})
            return
        reference = self._resume_options[index - 1]
        try:
            manifest = self._manifest_store.load_manifest(reference.session_id)
        except FileNotFoundError:
            self._emit("log", {"text": "セッションファイルが見つかりませんでした。"})
            return
        self._apply_manifest(manifest)
        self._emit("log", {"text": f"セッション {manifest.session_id} を読み込みました。"})
        if manifest.scoreboard:
            self._emit("scoreboard", {"scoreboard": manifest.scoreboard})
        self._show_conversation_log(manifest.conversation_log)

    def _apply_manifest(self, manifest: SessionManifest) -> None:
        self._config.session_id = manifest.session_id
        self._config.tmux_session = manifest.tmux_session
        self._config.worker_count = manifest.worker_count
        self._config.mode = SessionMode(manifest.mode)
        self._config.logs_root = Path(manifest.logs_dir).parent if manifest.logs_dir else Path("logs")
        self._config.reuse_existing_session = True
        self._last_scoreboard = manifest.scoreboard or {}
        self._last_instruction = manifest.latest_instruction
        self._last_selected_session = manifest.selected_session_id
        self._emit_status("再開準備完了")
        self._ensure_tmux_session(manifest)

    def _ensure_tmux_session(self, manifest: SessionManifest) -> None:
        try:
            import libtmux
        except ImportError:
            self._emit("log", {"text": "libtmux が見つかりません。tmux セッションは手動で復元してください。"})
            return

        server = libtmux.Server()  # type: ignore[attr-defined]
        existing = server.find_where({"session_name": manifest.tmux_session})
        if existing is not None:
            return

        worker_count = len(manifest.workers)
        orchestrator = self._builder(
            worker_count=worker_count,
            log_dir=Path(manifest.logs_dir) if manifest.logs_dir else None,
            session_name=manifest.tmux_session,
            reuse_existing_session=False,
        )
        tmux_manager = orchestrator._tmux  # type: ignore[attr-defined]
        orchestrator._worktree.prepare()  # type: ignore[attr-defined]
        boss_path = Path(manifest.boss.worktree) if manifest.boss and manifest.boss.worktree else self._worktree_root
        tmux_manager.set_boss_path(boss_path)
        tmux_manager.set_reuse_existing_session(True)
        layout = tmux_manager.ensure_layout(session_name=manifest.tmux_session, worker_count=worker_count)
        tmux_manager.resume_session(
            pane_id=layout.main_pane,
            workdir=self._worktree_root,
            session_id=manifest.main.session_id,
        )
        for idx, worker_name in enumerate(layout.worker_names):
            record = manifest.workers.get(worker_name)
            if not record or not record.worktree:
                continue
            pane_id = layout.worker_panes[idx]
            tmux_manager.resume_session(
                pane_id=pane_id,
                workdir=Path(record.worktree),
                session_id=record.session_id,
            )
        if manifest.boss and manifest.boss.worktree:
            tmux_manager.resume_session(
                pane_id=layout.boss_pane,
                workdir=Path(manifest.boss.worktree),
                session_id=manifest.boss.session_id,
            )

    def _show_conversation_log(self, log_path: Optional[str]) -> None:
        if not log_path:
            self._emit("log", {"text": "会話ログはありません。"})
            return
        path = Path(log_path)
        if not path.exists():
            self._emit("log", {"text": "会話ログが見つかりませんでした。"})
            return
        lines: List[str] = ["--- Conversation Log ---"]
        try:
            if path.suffix == ".jsonl":
                for raw_line in path.read_text(encoding="utf-8").splitlines():
                    if not raw_line.strip():
                        continue
                    data = json.loads(raw_line)
                    event_type = data.get("type")
                    if event_type == "instruction":
                        lines.append(f"[instruction] {data.get('instruction', '')}")
                    elif event_type == "fork":
                        workers = ", ".join(data.get("fork_map", {}).keys())
                        lines.append(f"[fork] workers={workers}")
                    elif event_type == "completion":
                        done = [k for k, v in (data.get("completion") or {}).items() if v.get("done")]
                        lines.append(f"[completion] done={done}")
                    elif event_type == "scoreboard":
                        lines.append("[scoreboard]")
                        for key, info in (data.get("scoreboard") or {}).items():
                            score = info.get("score")
                            selected = " [selected]" if info.get("selected") else ""
                            comment = info.get("comment", "")
                            lines.append(f"  {key}: {score} {selected} {comment}")
                    elif event_type == "selection":
                        lines.append(
                            f"[selection] session={data.get('selected_session')} key={data.get('selected_key')}"
                        )
                    elif event_type == "artifact":
                        workers = list((data.get("worker_sessions") or {}).keys())
                        lines.append(f"[artifact] main={data.get('main_session_id')} workers={workers}")
                    else:
                        lines.append(raw_line)
            else:
                lines.extend(path.read_text(encoding="utf-8").splitlines())
        except json.JSONDecodeError:
            lines.extend(path.read_text(encoding="utf-8").splitlines())
        lines.append("--- End Conversation Log ---")
        self._emit("log", {"text": "\n".join(lines)})

    def _emit_status(self, message: str) -> None:
        self._emit("status", {"message": message})

    def _emit(self, event_type: str, payload: Dict[str, object]) -> None:
        self._event_handler(event_type, payload)

    def _create_initial_config(self) -> SessionConfig:
        session_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + datetime.utcnow().strftime("%f")[:6]
        tmux_session = f"parallel-dev-{session_id}"
        logs_root = Path("logs") / session_id
        logs_root.mkdir(parents=True, exist_ok=True)
        return SessionConfig(
            session_id=session_id,
            tmux_session=tmux_session,
            worker_count=3,
            mode=SessionMode.PARALLEL,
            logs_root=logs_root,
        )

    def _create_cycle_logs_dir(self) -> Path:
        timestamp = datetime.utcnow().strftime("%y-%m-%d-%H%M%S")
        logs_dir = self._config.logs_root / timestamp
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir

    async def _wait_for_session(self, session_name: str, attempts: int = 20, delay: float = 0.25) -> bool:
        for _ in range(attempts):
            if self._attach_manager.session_exists(session_name):
                return True
            await asyncio.sleep(delay)
        return False

    def _load_settings(self) -> Dict[str, object]:
        if not self._settings_path.exists():
            return {}
        try:
            return json.loads(self._settings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_settings(self) -> None:
        data = dict(self._settings)
        data["attach_mode"] = self._attach_mode
        try:
            self._settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._settings = data
        except OSError:
            pass

    def _build_manifest(self, result: OrchestrationResult, logs_dir: Path) -> SessionManifest:
        assert result.artifact is not None
        artifact = result.artifact
        main_record = PaneRecord(
            role="main",
            name=None,
            session_id=artifact.main_session_id,
            worktree=str(self._worktree_root),
        )
        workers = {
            name: PaneRecord(
                role="worker",
                name=name,
                session_id=session_id,
                worktree=str(artifact.worker_paths.get(name)) if artifact.worker_paths.get(name) else None,
            )
            for name, session_id in artifact.worker_sessions.items()
        }
        boss_record = (
            PaneRecord(
                role="boss",
                name="boss",
                session_id=artifact.boss_session_id,
                worktree=str(artifact.boss_path) if artifact.boss_path else None,
            )
            if artifact.boss_session_id
            else None
        )
        conversation_path = None
        if artifact.log_paths:
            conversation_path = str(artifact.log_paths.get("jsonl") or artifact.log_paths.get("yaml"))
        else:
            conversation_path = str(logs_dir / "instruction.log")
        return SessionManifest(
            session_id=self._config.session_id,
            created_at=datetime.utcnow().isoformat(timespec="seconds"),
            tmux_session=self._config.tmux_session,
            worker_count=len(workers),
            mode=self._config.mode.value,
            logs_dir=str(logs_dir),
            latest_instruction=self._last_instruction,
            scoreboard=self._last_scoreboard,
            conversation_log=conversation_path,
            selected_session_id=artifact.selected_session_id,
            main=main_record,
            boss=boss_record,
            workers=workers,
        )

    @staticmethod
    def _default_builder(
        *,
        worker_count: int,
        log_dir: Optional[Path],
        session_name: Optional[str] = None,
        reuse_existing_session: bool = False,
    ) -> Orchestrator:
        raise RuntimeError("Orchestrator builder is not configured.")


class ParallelDeveloperApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        layout: vertical;
        height: 1fr;
        padding: 1 2;
    }

    #log {
        height: 1fr;
        border: round $success;
        margin-bottom: 1;
        overflow-x: hidden;
    }

    #log.paused {
        border: round $warning;
    }

    #status {
        border: round $success;
        padding: 1;
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
        ("tab", "palette_next", "次候補"),
        ("shift+tab", "palette_previous", "前候補"),
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
            with Vertical():
                self.status_panel = StatusPanel(id="status")
                yield self.status_panel
                self.log_panel = EventLog(id="log", max_lines=400)
                yield self.log_panel
                self.selection_list = OptionList(id="selection")
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
        value = self.command_input.text
        if value == self._last_command_text:
            return
        if not value:
            self._last_command_text = value
            self._hide_command_palette()
            return
        if not value.startswith("/"):
            self._last_command_text = value
            self._hide_command_palette()
            return
        command, has_space, remainder = value.partition(" ")
        command = command.lower()
        if not has_space:
            self._pending_command = None
            self._update_command_suggestions(command)
            self._last_command_text = value
            return
        spec = self.controller._command_specs.get(command)
        if spec is None:
            self._last_command_text = value
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
            label = opt.label
            value_str = str(opt.value)
            if not remainder or value_str.startswith(remainder) or label.lower().startswith(remainder.lower()):
                filtered.append(PaletteItem(label, opt.value))
        if not filtered:
            self._last_command_text = value
            self._hide_command_palette()
            return
        self._pending_command = command
        self._show_command_palette(filtered, mode="options")
        self._last_command_text = value

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
        if event.event_type == "status" and self.status_panel:
            message = event.payload.get("message", "")
            self.status_panel.update_status(self.controller._config, str(message))
        elif event.event_type == "log" and self.log_panel:
            text = str(event.payload.get("text", ""))
            self.log_panel.log(text)
        elif event.event_type == "scoreboard":
            scoreboard = event.payload.get("scoreboard", {})
            if isinstance(scoreboard, dict):
                self._render_scoreboard(scoreboard)
        elif event.event_type == "log_copy":
            message = self._copy_log_to_clipboard()
            self._notify_status(message)
        elif event.event_type == "log_save":
            destination = str(event.payload.get("path", "") or "").strip()
            if not destination:
                self._notify_status("保存先パスが指定されていません。")
            else:
                message = self._save_log_to_path(destination)
                self._notify_status(message)
        elif event.event_type == "pause_state":
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
        elif event.event_type == "selection_request":
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
            if self.command_input:
                self.command_input.display = False
        elif event.event_type == "selection_finished":
            if self.selection_list:
                self.selection_list.display = False
            if self.command_input:
                self.command_input.display = True
                self.command_input.focus()
        elif event.event_type == "quit":
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

    def _update_command_suggestions(self, prefix: str) -> None:
        suggestions = self.controller.get_command_suggestions(prefix)
        if not suggestions:
            self._hide_command_palette()
            return
        items = [PaletteItem(f"{s.name:<10} {s.description}", s.name) for s in suggestions]
        self._show_command_palette(items, mode="command")

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
        if from_selection:
            return "選択範囲をクリップボードへコピーしました。"
        return "ログ全体をクリップボードへコピーしました。"

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

    def on_key(self, event: events.Key) -> None:
        if self._handle_ctrl_c(event):
            return
        if self._handle_text_shortcuts(event):
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

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
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


def build_orchestrator(
    *,
    worker_count: int,
    log_dir: Optional[Path],
    session_name: Optional[str] = None,
    reuse_existing_session: bool = False,
) -> Orchestrator:
    session_name = session_name or "parallel-dev"
    timestamp = datetime.utcnow().strftime("%y-%m-%d-%H%M%S")
    base_logs_dir = Path(log_dir) if log_dir else Path("logs") / timestamp
    base_logs_dir.mkdir(parents=True, exist_ok=True)
    session_map_root = base_logs_dir.parent if log_dir else base_logs_dir
    session_map_root.mkdir(parents=True, exist_ok=True)
    session_map_path = session_map_root / "sessions_map.yaml"

    monitor = CodexMonitor(logs_dir=base_logs_dir, session_map_path=session_map_path)
    worktree_root = Path.cwd()
    tmux_manager = TmuxLayoutManager(
        session_name=session_name,
        worker_count=worker_count,
        monitor=monitor,
        root_path=worktree_root,
        startup_delay=0.5,
        backtrack_delay=0.3,
        reuse_existing_session=reuse_existing_session,
    )
    worktree_manager = WorktreeManager(root=worktree_root, worker_count=worker_count)
    log_manager = LogManager(logs_dir=base_logs_dir)

    return Orchestrator(
        tmux_manager=tmux_manager,
        worktree_manager=worktree_manager,
        monitor=monitor,
        log_manager=log_manager,
        worker_count=worker_count,
        session_name=session_name,
    )


def run() -> None:
    ParallelDeveloperApp().run()
