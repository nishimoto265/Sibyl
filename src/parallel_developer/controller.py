"""Controller and orchestration helpers for the Parallel Developer CLI."""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Mapping, Awaitable
import platform
import subprocess
import shlex
import shutil
from subprocess import PIPE

from .orchestrator import BossMode, CandidateInfo, CycleLayout, OrchestrationResult, Orchestrator, SelectionDecision
from .session_manifest import ManifestStore, PaneRecord, SessionManifest, SessionReference
from .services import CodexMonitor, LogManager, TmuxLayoutManager, WorktreeManager
from .settings_store import SettingsStore
from .workflow import WorkflowManager


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
    boss_mode: BossMode = BossMode.SCORE
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


@dataclass
class CommandSpecEntry:
    description: str
    handler: Callable[[Optional[object]], Awaitable[None]]
    options: Optional[List[CommandOption]] = None
    options_provider: Optional[Callable[[], List[CommandOption]]] = None


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
        settings_path = Path(settings_path) if settings_path else (self._worktree_root / ".parallel-dev" / "settings.json")
        self._settings_store = SettingsStore(settings_path)
        self._attach_mode: str = self._settings_store.attach_mode
        self._codex_home_mode: str = self._settings_store.codex_home_mode
        saved_boss_mode = self._settings_store.boss_mode
        try:
            self._config.boss_mode = BossMode(saved_boss_mode)
        except ValueError:
            self._config.boss_mode = BossMode.SCORE
        self._session_namespace: str = self._config.session_id
        self._session_root: Path = self._worktree_root / ".parallel-dev" / "sessions" / self._session_namespace
        self._codex_home: Path = self._session_root / "codex-home"
        self._command_specs: Dict[str, CommandSpecEntry] = self._build_command_specs()
        self._workflow = WorkflowManager(self)

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
                suggestions.append(CommandSuggestion(name=name, description=spec.description))
        if not suggestions and prefix == "/":
            for name in sorted(self._command_specs.keys()):
                spec = self._command_specs[name]
                suggestions.append(CommandSuggestion(name=name, description=spec.description))
        return suggestions

    def get_command_options(self, name: str) -> List[CommandOption]:
        spec = self._command_specs.get(name)
        if not spec:
            return []
        if spec.options_provider:
            return spec.options_provider()
        if spec.options:
            return list(spec.options)
        return []

    async def execute_command(self, name: str, option: Optional[object] = None) -> None:
        spec = self._command_specs.get(name)
        if spec is None:
            self._emit("log", {"text": f"未知のコマンドです: {name}"})
            return
        await spec.handler(option)

    def _build_command_specs(self) -> Dict[str, CommandSpecEntry]:
        return {
            "/attach": CommandSpecEntry(
                "tmux セッションへの接続モードを切り替える、または即座に接続する",
                self._cmd_attach,
                options=[
                    CommandOption("auto", "auto"),
                    CommandOption("manual", "manual"),
                    CommandOption("now", "now"),
                ],
            ),
            "/boss": CommandSpecEntry(
                "Boss モードを切り替える",
                self._cmd_boss,
                options=[
                    CommandOption("skip", "skip"),
                    CommandOption("score", "score"),
                    CommandOption("rewrite", "rewrite"),
                ],
            ),
            "/codexhome": CommandSpecEntry(
                "Codex HOME のモードを切り替える (session/shared)",
                self._cmd_codex_home,
            ),
            "/parallel": CommandSpecEntry(
                "ワーカー数を設定する",
                self._cmd_parallel,
                options=[CommandOption(str(n), str(n)) for n in range(1, 5)],
            ),
            "/mode": CommandSpecEntry(
                "実行モードを切り替える",
                self._cmd_mode,
                options=[CommandOption("main", "main"), CommandOption("parallel", "parallel")],
            ),
            "/resume": CommandSpecEntry(
                "保存済みセッションを再開する",
                self._cmd_resume,
                options_provider=self._build_resume_options,
            ),
            "/continue": CommandSpecEntry(
                "現行サイクルでワーカーへの追加指示を続ける",
                self._cmd_continue,
            ),
            "/log": CommandSpecEntry(
                "ログをコピーするかファイルへ保存する",
                self._cmd_log,
                options=[
                    CommandOption("copy", "copy"),
                    CommandOption("save", "save"),
                ],
            ),
            "/status": CommandSpecEntry(
                "現在の状態を表示する",
                self._cmd_status,
            ),
            "/scoreboard": CommandSpecEntry(
                "直近のスコアボードを表示する",
                self._cmd_scoreboard,
            ),
            "/done": CommandSpecEntry(
                "全ワーカーに /done を送信して採点フェーズへ移行する",
                self._cmd_done,
            ),
            "/help": CommandSpecEntry(
                "コマンド一覧を表示する",
                self._cmd_help,
            ),
            "/exit": CommandSpecEntry(
                "CLI を終了する",
                self._cmd_exit,
            ),
        }

    async def _cmd_exit(self, option: Optional[object]) -> None:
        self._emit("quit", {})

    async def _cmd_help(self, option: Optional[object]) -> None:
        lines = ["利用可能なコマンド:"]
        for name in sorted(self._command_specs.keys()):
            spec = self._command_specs[name]
            lines.append(f"  {name:10s} : {spec.description}")
        self._emit("log", {"text": "\n".join(lines)})

    async def _cmd_status(self, option: Optional[object]) -> None:
        self._emit_status("待機中")

    async def _cmd_scoreboard(self, option: Optional[object]) -> None:
        self._emit("scoreboard", {"scoreboard": self._last_scoreboard})

    async def _cmd_done(self, option: Optional[object]) -> None:
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

    async def _cmd_continue(self, option: Optional[object]) -> None:
        if self._continue_future and not self._continue_future.done():
            self._continue_future.set_result(True)
            self._emit("log", {"text": "/continue を受け付けました。ワーカーに追加指示を送れます。"})
        else:
            self._emit("log", {"text": "/continue は現在利用できません。"})

    async def _cmd_boss(self, option: Optional[object]) -> None:
        if option is None:
            mode = self._config.boss_mode.value
            self._emit(
                "log",
                {
                    "text": (
                        "現在の Boss モードは {mode} です。"
                        " (skip=採点スキップ, score=採点のみ, rewrite=再実装)"
                    ).format(mode=mode)
                },
            )
            return
        value = str(option).lower()
        mapping = {
            "skip": BossMode.SKIP,
            "score": BossMode.SCORE,
            "rewrite": BossMode.REWRITE,
        }
        new_mode = mapping.get(value)
        if new_mode is None:
            self._emit("log", {"text": "使い方: /boss skip | /boss score | /boss rewrite"})
            return
        if new_mode == self._config.boss_mode:
            self._emit("log", {"text": f"Boss モードは既に {new_mode.value} です。"})
            return
        self._config.boss_mode = new_mode
        self._settings_store.boss_mode = new_mode.value
        self._emit("log", {"text": f"Boss モードを {new_mode.value} に設定しました。"})

    async def _cmd_codex_home(self, option: Optional[object]) -> None:
        env_override = os.getenv("PARALLEL_DEV_CODEX_HOME_MODE")
        if env_override:
            self._emit(
                "log",
                {
                    "text": (
                        "環境変数 PARALLEL_DEV_CODEX_HOME_MODE が設定されているため、"
                        "/codexhome での変更は無効です。"
                    )
                },
            )
            return
        if option is None:
            self._emit(
                "log",
                {"text": f"現在の Codex HOME モードは {self._codex_home_mode} です。（session/shared）"},
            )
            return
        mode = str(option).lower()
        if mode not in {"session", "shared"}:
            self._emit("log", {"text": "使い方: /codexhome session | /codexhome shared"})
            return
        if mode == self._codex_home_mode:
            self._emit("log", {"text": f"Codex HOME モードは既に {mode} です。"})
            return
        self._codex_home_mode = mode
        self._settings_store.codex_home_mode = mode
        self._emit("log", {"text": f"Codex HOME モードを {mode} に設定しました。次のサイクルから適用されます。"})

    async def _cmd_attach(self, option: Optional[object]) -> None:
        mode = str(option).lower() if option is not None else None
        if mode in {"auto", "manual"}:
            self._attach_mode = mode
            self._emit("log", {"text": f"/attach モードを {mode} に設定しました。"})
            self._settings_store.attach_mode = mode
            return
        if mode == "now" or option is None:
            await self._handle_attach_command(force=True)
            return
        self._emit("log", {"text": "使い方: /attach [auto|manual|now]"})

    async def _cmd_parallel(self, option: Optional[object]) -> None:
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

    async def _cmd_mode(self, option: Optional[object]) -> None:
        mode = str(option).lower() if option is not None else None
        if mode not in {"main", "parallel"}:
            self._emit("log", {"text": "使い方: /mode main | /mode parallel"})
            return
        self._config.mode = SessionMode(mode)
        self._emit_status("設定を更新しました。")

    async def _cmd_resume(self, option: Optional[object]) -> None:
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

    async def _cmd_log(self, option: Optional[object]) -> None:
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

    def _request_selection(self, candidates: List[CandidateInfo], scoreboard: Optional[Dict[str, Dict[str, object]]] = None) -> Future:
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
        return future

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
        await self._workflow.run_instruction(instruction)

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
            boss_mode=BossMode.SCORE,
        )

    def _create_cycle_logs_dir(self) -> Path:
        timestamp = datetime.utcnow().strftime("%y-%m-%d-%H%M%S")
        logs_dir = self._config.logs_root / timestamp
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir

    def _ensure_codex_home(self) -> Path:
        codex_home, mode = self._resolve_codex_home()
        codex_home.mkdir(parents=True, exist_ok=True)
        if mode == "session":
            self._session_root.mkdir(parents=True, exist_ok=True)
            self._bootstrap_codex_home(codex_home)
        sessions_dir = codex_home / ".codex" / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)
        self._codex_home = codex_home
        return codex_home

    def _resolve_codex_home(self) -> tuple[Path, str]:
        env_mode = os.getenv("PARALLEL_DEV_CODEX_HOME_MODE", "").lower()
        mode = env_mode if env_mode in {"session", "shared"} else self._codex_home_mode

        env_path_raw = os.getenv("PARALLEL_DEV_CODEX_HOME")
        if env_path_raw:
            expanded = os.path.expandvars(os.path.expanduser(env_path_raw))
            if "{session}" in expanded:
                resolved = expanded.replace("{session}", self._session_namespace)
                return Path(resolved), mode
            path = Path(expanded)
            if mode == "session" and env_mode == "":
                return path / self._session_namespace, mode
            return path, mode

        if mode == "shared":
            return Path.home(), mode
        return self._session_root / "codex-home", mode

    def _bootstrap_codex_home(self, codex_home: Path) -> None:
        global_codex = Path.home() / ".codex"
        if codex_home == Path.home():
            return
        target_codex = codex_home / ".codex"
        sentinel = target_codex / ".bootstrap_complete"
        if sentinel.exists():
            return
        if not global_codex.exists():
            return
        target_codex.mkdir(parents=True, exist_ok=True)
        for item in global_codex.iterdir():
            if item.name == "sessions":
                continue
            destination = target_codex / item.name
            if destination.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, destination, dirs_exist_ok=True)
            else:
                try:
                    shutil.copy2(item, destination)
                except OSError:
                    continue
        try:
            sentinel.write_text(datetime.utcnow().isoformat(timespec="seconds"), encoding="utf-8")
        except OSError:
            pass

    async def _wait_for_session(self, session_name: str, attempts: int = 20, delay: float = 0.25) -> bool:
        for _ in range(attempts):
            if self._attach_manager.session_exists(session_name):
                return True
            await asyncio.sleep(delay)
        return False

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


def build_orchestrator(
    *,
    worker_count: int,
    log_dir: Optional[Path],
    session_name: Optional[str] = None,
    reuse_existing_session: bool = False,
    session_namespace: Optional[str] = None,
    codex_home: Optional[Path] = None,
    boss_mode: BossMode = BossMode.SCORE,
) -> Orchestrator:
    session_name = session_name or "parallel-dev"
    timestamp = datetime.utcnow().strftime("%y-%m-%d-%H%M%S")
    base_logs_dir = Path(log_dir) if log_dir else Path("logs") / timestamp
    base_logs_dir.mkdir(parents=True, exist_ok=True)
    session_map_root = base_logs_dir.parent if log_dir else base_logs_dir
    session_map_root.mkdir(parents=True, exist_ok=True)
    session_map_path = session_map_root / "sessions_map.yaml"

    worktree_root = Path.cwd()
    session_root = worktree_root / ".parallel-dev"
    if session_namespace:
        session_root = session_root / "sessions" / session_namespace
    session_root.mkdir(parents=True, exist_ok=True)

    codex_home = Path(codex_home) if codex_home else session_root / "codex-home"
    codex_home.mkdir(parents=True, exist_ok=True)
    codex_sessions_root = codex_home / ".codex" / "sessions"
    codex_sessions_root.mkdir(parents=True, exist_ok=True)

    monitor = CodexMonitor(
        logs_dir=base_logs_dir,
        session_map_path=session_map_path,
        codex_sessions_root=codex_sessions_root,
    )
    tmux_manager = TmuxLayoutManager(
        session_name=session_name,
        worker_count=worker_count,
        monitor=monitor,
        root_path=worktree_root,
        startup_delay=0.5,
        backtrack_delay=0.3,
        reuse_existing_session=reuse_existing_session,
        session_namespace=session_namespace,
        codex_home=codex_home,
    )
    worktree_manager = WorktreeManager(
        root=worktree_root,
        worker_count=worker_count,
        session_namespace=session_namespace,
    )
    log_manager = LogManager(logs_dir=base_logs_dir)

    return Orchestrator(
        tmux_manager=tmux_manager,
        worktree_manager=worktree_manager,
        monitor=monitor,
        log_manager=log_manager,
        worker_count=worker_count,
        session_name=session_name,
        boss_mode=boss_mode,
    )
