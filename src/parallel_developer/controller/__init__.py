"""Controller and orchestration helpers for the Sibyl CLI."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Mapping, Awaitable, Union
import platform
import subprocess
import shlex
import shutil
from subprocess import PIPE

import git

from .commands import CommandOption, CommandSpecEntry, CommandSuggestion, build_command_specs
from .events import ControllerEventType
from .flow import WorkerFlowHelper
from .pause import PauseHelper
from .history import HistoryManager
from ..orchestrator import (
    BossMode,
    CandidateInfo,
    CycleLayout,
    OrchestrationResult,
    Orchestrator,
    SelectionDecision,
    WorkerDecision,
    MergeMode,
    MergeOutcome,
)
from ..services import CodexMonitor, LogManager, TmuxLayoutManager, WorktreeManager
from ..stores import (
    ManifestStore,
    PaneRecord,
    SessionManifest,
    SessionReference,
    SettingsStore,
    default_config_dir,
    resolve_settings_path,
    resolve_worktree_root,
)
from .workflow_runner import WorkflowRunner

def _ensure_logs_directory(identifier: str) -> Path:
    """Create and return the logs directory for the given identifier."""
    base_dir = default_config_dir() / "logs"
    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / identifier
    target.mkdir(parents=True, exist_ok=True)
    return target


class SessionMode(str, Enum):
    PARALLEL = "parallel"
    MAIN = "main"


class FlowMode(str, Enum):
    MANUAL = "manual"
    AUTO_REVIEW = "auto_review"
    AUTO_SELECT = "auto_select"
    FULL_AUTO = "full_auto"


FLOW_MODE_LABELS = {
    FlowMode.MANUAL: "Manual",
    FlowMode.AUTO_REVIEW: "Auto Review",
    FlowMode.AUTO_SELECT: "Auto Select",
    FlowMode.FULL_AUTO: "Full Auto",
}
MERGE_STRATEGY_LABELS = {
    MergeMode.MANUAL: "Manual",
    MergeMode.AUTO: "Auto",
}


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
    flow_mode: FlowMode = FlowMode.MANUAL
    reuse_existing_session: bool = False
    merge_mode: MergeMode = MergeMode.MANUAL

@dataclass
class SelectionContext:
    future: Future
    candidates: List[CandidateInfo]
    scoreboard: Dict[str, Dict[str, object]]


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
        self._history = HistoryManager()
        self._cycle_counter: int = 0
        self._current_cycle_id: Optional[int] = None
        self._cancelled_cycles: Set[int] = set()
        self._last_tmux_manager: Optional[TmuxLayoutManager] = None
        self._active_orchestrator: Optional[Orchestrator] = None
        self._queued_instruction: Optional[str] = None
        self._continue_future: Optional[Future] = None
        self._continuation_input_future: Optional[Future] = None
        self._awaiting_continuation_input: bool = False
        self._attach_manager = TmuxAttachManager()
        explicit_settings_path = Path(settings_path).expanduser() if settings_path else None
        resolved_settings_path = resolve_settings_path(explicit_settings_path)
        self._settings_store = SettingsStore(resolved_settings_path)
        self._worktree_storage_root = resolve_worktree_root(
            self._settings_store.worktree_root,
            self._worktree_root,
        )
        self._attach_mode: str = self._settings_store.attach
        saved_boss_mode = self._settings_store.boss
        try:
            self._config.boss_mode = BossMode(saved_boss_mode)
        except ValueError:
            self._config.boss_mode = BossMode.SCORE
        saved_flow_mode = self._settings_store.flow
        try:
            self._flow_mode = FlowMode(saved_flow_mode)
        except ValueError:
            self._flow_mode = FlowMode.FULL_AUTO
        self._config.flow_mode = self._flow_mode
        saved_merge_mode = self._settings_store.merge
        try:
            self._merge_mode = MergeMode(saved_merge_mode)
        except ValueError:
            self._merge_mode = MergeMode.AUTO
        self._config.merge_mode = self._merge_mode
        self._auto_commit_enabled: bool = self._settings_store.commit == "auto"
        try:
            self._config.worker_count = max(1, int(self._settings_store.parallel or "3"))
        except ValueError:
            self._config.worker_count = 3
        try:
            self._config.mode = SessionMode(self._settings_store.mode)
        except ValueError:
            self._config.mode = SessionMode.PARALLEL
        self._session_namespace: str = self._config.session_id
        self._last_started_main_session_id: Optional[str] = None
        self._pre_cycle_selected_session: Optional[str] = None
        self._pre_cycle_selected_session_set: bool = False
        self._command_specs: Dict[str, CommandSpecEntry] = build_command_specs(
            self,
            flow_mode_cls=FlowMode,
            boss_mode_cls=BossMode,
            merge_mode_cls=MergeMode,
        )
        self._worker_flow = WorkerFlowHelper(self, FlowMode)
        self._pause_helper = PauseHelper(self)
        self._log_hook = self._handle_orchestrator_log
        self._workflow = WorkflowRunner(self)

    async def handle_input(self, user_input: str) -> None:
        raw_text = user_input.rstrip("\n")

        if self._awaiting_continuation_input:
            stripped = raw_text.strip()
            if not stripped:
                self._emit(ControllerEventType.LOG, {"text": "追加指示が空です。何をするか具体的に入力してください。"})
                return
            if stripped.startswith("/"):
                self._emit(ControllerEventType.LOG, {"text": "追加指示入力中です。コマンドではなくテキストで指示を入力してください。"})
                return
            self._submit_continuation_instruction(raw_text)
            return

        text = raw_text.strip()
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
                self._emit(ControllerEventType.LOG, {"text": "キャンセル処理中です。完了後にこの指示を実行します。"})
                return
        if self._running:
            self._emit(ControllerEventType.LOG, {"text": "別の指示を処理中です。完了を待ってから再度実行してください。"})
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
        if name == "/marge":
            name = "/merge"
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
        options: List[CommandOption] = []
        if spec.options_provider:
            options = spec.options_provider()
        elif spec.options:
            options = list(spec.options)
        return options

    async def execute_command(self, name: str, option: Optional[object] = None) -> None:
        spec = self._command_specs.get(name)
        if spec is None:
            self._emit(ControllerEventType.LOG, {"text": f"未知のコマンドです: {name}"})
            return
        await spec.handler(option)

    async def _cmd_exit(self, option: Optional[object]) -> None:
        self._emit(ControllerEventType.QUIT, {})

    async def _cmd_help(self, option: Optional[object]) -> None:
        lines = ["利用可能なコマンド:"]
        for name in sorted(self._command_specs.keys()):
            spec = self._command_specs[name]
            lines.append(f"  {name:10s} : {spec.description}")
        self._emit(ControllerEventType.LOG, {"text": "\n".join(lines)})

    async def _cmd_status(self, option: Optional[object]) -> None:
        self._emit_status("待機中")

    async def _cmd_scoreboard(self, option: Optional[object]) -> None:
        self._emit(ControllerEventType.SCOREBOARD, {"scoreboard": self._last_scoreboard})

    async def _cmd_done(self, option: Optional[object]) -> None:
        if self._continue_future and not self._continue_future.done():
            self._continue_future.set_result("done")
            self._emit(ControllerEventType.LOG, {"text": "/done を受け付けました。採点フェーズへ移行します。"})
            return
        if self._active_orchestrator:
            count = self._active_orchestrator.force_complete_workers()
            if count:
                self._emit(ControllerEventType.LOG, {"text": f"/done を検知として扱い、{count} ワーカーを完了済みに設定しました。"})
            else:
                self._emit(ControllerEventType.LOG, {"text": "完了扱いにするワーカーセッションが見つかりませんでした。"})
        else:
            self._emit(ControllerEventType.LOG, {"text": "現在進行中のワーカークセッションがないため /done を適用できません。"})

    async def _cmd_continue(self, option: Optional[object]) -> None:
        if self._continue_future and not self._continue_future.done():
            self._continue_future.set_result("continue")
            self._emit(ControllerEventType.LOG, {"text": "/continue を受け付けました。追加指示を入力してください。"})
        else:
            self._emit(ControllerEventType.LOG, {"text": "/continue は現在利用できません。"})

    def _await_worker_command(self) -> str:
        future = Future()
        self._continue_future = future
        self._emit(
            ControllerEventType.LOG,
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
        return str(decision)

    def _await_continuation_instruction(self) -> str:
        future = Future()
        self._continuation_input_future = future
        self._awaiting_continuation_input = True
        self._emit(
            ControllerEventType.LOG,
            {"text": "追加指示を入力してください。完了したらワーカーのフラグ更新を待ちます。"},
        )
        try:
            instruction = future.result()
        finally:
            self._continuation_input_future = None
            self._awaiting_continuation_input = False
        return str(instruction).strip()

    def _submit_continuation_instruction(self, instruction: str) -> None:
        future = self._continuation_input_future
        self._continuation_input_future = None
        self._awaiting_continuation_input = False
        if future and not future.done():
            future.set_result(instruction.strip())
            self._emit(ControllerEventType.LOG, {"text": "追加指示を受け付けました。ワーカーの完了を待ちます。"})

    async def _cmd_boss(self, option: Optional[object]) -> None:
        if option is None:
            mode = self._config.boss_mode.value
            self._emit(
                ControllerEventType.LOG,
                {
                    "text": (
                        "現在の Boss モードは {mode} です。"
                        " (skip=採点スキップ, score=採点のみ, rewrite=再実装)"
                    ).format(mode=mode)
                },
            )
            return
        value = str(option).strip()
        mapping = {
            "skip": BossMode.SKIP,
            "score": BossMode.SCORE,
            "rewrite": BossMode.REWRITE,
        }
        new_mode = mapping.get(value.lower())
        if new_mode is None:
            # Treat unknown arguments as regular instructions for compatibility
            await self.handle_input(f"{value}")
            return
        if new_mode == self._config.boss_mode:
            self._emit(ControllerEventType.LOG, {"text": f"Boss モードは既に {new_mode.value} です。"})
            return
        self._config.boss_mode = new_mode
        self._settings_store.boss = new_mode.value
        self._emit(ControllerEventType.LOG, {"text": f"Boss モードを {new_mode.value} に設定しました。"})

    async def _cmd_flow(self, option: Optional[object]) -> None:
        if option is None:
            lines = [
                f"現在のフローモード: {self._flow_mode_display()}",
                "利用可能なモード:",
            ]
            for mode in FlowMode:
                lines.append(f"  {mode.value:12s} : {self._flow_mode_display(mode)}")
            lines.append("使い方: /flow [manual|auto_review|auto_select|full_auto]")
            self._emit(ControllerEventType.LOG, {"text": "\n".join(lines)})
            return

        token = str(option).strip().lower().replace("-", "_")
        mapping = {mode.value: mode for mode in FlowMode}
        new_mode = mapping.get(token)
        if new_mode is None:
            self._emit(
                ControllerEventType.LOG,
                {"text": "使い方: /flow [manual|auto_review|auto_select|full_auto]"},
            )
            return
        if new_mode == getattr(self, "_flow_mode", FlowMode.MANUAL):
            self._emit(ControllerEventType.LOG, {"text": f"フローモードは既に {self._flow_mode_display(new_mode)} です。"})
            return

        self._flow_mode = new_mode
        self._config.flow_mode = new_mode
        self._settings_store.flow = new_mode.value
        self._emit(ControllerEventType.LOG, {"text": f"フローモードを {self._flow_mode_display(new_mode)} に設定しました。"})
        self._emit_status("待機中")

    async def _cmd_merge(self, option: Optional[object]) -> None:
        if option is None:
            lines = [
                f"現在のマージ戦略: {self._merge_strategy_display()}",
                "利用可能な戦略:",
            ]
            for strategy in MergeMode:
                lines.append(f"  {strategy.value:16s} : {self._merge_strategy_display(strategy)}")
            lines.append("使い方: /merge [manual|auto]")
            self._emit(ControllerEventType.LOG, {"text": "\n".join(lines)})
            return

        token = str(option).strip().lower().replace("-", "_")
        mapping = {strategy.value: strategy for strategy in MergeMode}
        new_strategy = mapping.get(token)
        if new_strategy is None:
            self._emit(ControllerEventType.LOG, {"text": "使い方: /merge [manual|auto]"})
            return
        if new_strategy == getattr(self, "_merge_mode", MergeMode.MANUAL):
            self._emit(ControllerEventType.LOG, {"text": f"マージ戦略は既に {self._merge_strategy_display(new_strategy)} です。"})
            return

        self._merge_mode = new_strategy
        self._config.merge_mode = new_strategy
        self._settings_store.merge = new_strategy.value
        self._emit(ControllerEventType.LOG, {"text": f"マージ戦略を {self._merge_strategy_display(new_strategy)} に設定しました。"})
        self._emit_status("待機中")

    async def _cmd_attach(self, option: Optional[object]) -> None:
        mode = str(option).lower() if option is not None else None
        if mode in {"auto", "manual"}:
            self._attach_mode = mode
            self._emit(ControllerEventType.LOG, {"text": f"/attach モードを {mode} に設定しました。"})
            self._settings_store.attach = mode
            return
        if mode == "now" or option is None:
            await self._handle_attach_command(force=True)
            return
        self._emit(ControllerEventType.LOG, {"text": "使い方: /attach [auto|manual|now]"})

    async def _cmd_parallel(self, option: Optional[object]) -> None:
        if option is None:
            self._emit(ControllerEventType.LOG, {"text": "使い方: /parallel <ワーカー数>"})
            return
        try:
            value = int(str(option))
        except ValueError:
            self._emit(ControllerEventType.LOG, {"text": "ワーカー数は数字で指定してください。"})
            return
        if value < 1:
            self._emit(ControllerEventType.LOG, {"text": "ワーカー数は1以上で指定してください。"})
            return
        self._config.worker_count = value
        self._settings_store.parallel = str(value)
        self._emit_status("設定を更新しました。")

    async def _cmd_mode(self, option: Optional[object]) -> None:
        mode = str(option).lower() if option is not None else None
        if mode not in {"main", "parallel"}:
            self._emit(ControllerEventType.LOG, {"text": "使い方: /mode main | /mode parallel"})
            return
        self._config.mode = SessionMode(mode)
        self._settings_store.mode = mode
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
            self._emit(ControllerEventType.LOG, {"text": "指定されたセッションが見つかりません。"})
            return
        self._load_session(index)

    async def _cmd_log(self, option: Optional[object]) -> None:
        if option is None:
            self._emit(
                ControllerEventType.LOG,
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
            self._emit(ControllerEventType.LOG_COPY, {})
            return
        if action == "save":
            if not argument:
                self._emit(ControllerEventType.LOG, {"text": "保存先パスを指定してください。例: /log save logs/output.log"})
                return
            self._emit(ControllerEventType.LOG_SAVE, {"path": argument})
            return
        self._emit(ControllerEventType.LOG, {"text": "使い方: /log copy | /log save <path>"})

    async def _cmd_commit(self, option: Optional[object]) -> None:
        mode = "manual"
        if option is not None:
            mode = str(option).lower()
        if mode == "manual":
            self._auto_commit_enabled = False
            self._settings_store.commit = "manual"
            self._perform_commit(auto=False, quiet_when_no_change=False)
            return
        if mode == "auto":
            if self._auto_commit_enabled:
                self._auto_commit_enabled = False
                self._settings_store.commit = "manual"
                self._emit(ControllerEventType.LOG, {"text": "自動コミットを無効にしました。"})
            else:
                self._auto_commit_enabled = True
                self._settings_store.commit = "auto"
                self._emit(ControllerEventType.LOG, {"text": "自動コミットを有効にしました。"})
                self._perform_commit(auto=True, quiet_when_no_change=True)
            return
        self._emit(ControllerEventType.LOG, {"text": "/commit は manual または auto を指定してください。"})

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
            self._emit(ControllerEventType.LOG, {"text": f"tmuxセッション {session_name} にペインが見つかりませんでした。"})
            return

        for pane_id in pane_ids:
            subprocess.run(
                ["tmux", "send-keys", "-t", pane_id, "Escape"],
                check=False,
            )
        self._emit(ControllerEventType.LOG, {"text": f"tmuxセッション {session_name} の {len(pane_ids)} 個のペインへEscapeを送信しました。"})

    def handle_escape(self) -> None:
        self._pause_helper.handle_escape()
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
            self._emit(ControllerEventType.LOG, {"text": "tmux コマンドが見つかりません。tmuxがインストールされているか確認してください。"})
            return None
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            if message:
                self._emit(ControllerEventType.LOG, {"text": f"tmux list-panes に失敗しました: {message}"})
            return None
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]

    async def _dispatch_paused_instruction(self, instruction: str) -> None:
        await self._pause_helper.dispatch_paused_instruction(instruction)

    def _record_cycle_snapshot(self, result: OrchestrationResult, cycle_id: int) -> None:
        snapshot = {
            "cycle_id": cycle_id,
            "selected_session": result.selected_session,
            "scoreboard": dict(result.sessions_summary),
            "instruction": self._last_instruction,
        }
        self._history.record_cycle_snapshot(result, cycle_id, self._last_instruction)

    def _handle_worker_decision(
        self,
        fork_map: Mapping[str, str],
        completion_info: Mapping[str, Any],
        layout: CycleLayout,
    ) -> WorkerDecision:
        return self._worker_flow.handle_worker_decision(fork_map, completion_info, layout)

    def _record_history(self, text: str) -> None:
        self._history.record_input(text)

    def _perform_commit(self, *, auto: bool, quiet_when_no_change: bool) -> bool:
        try:
            repo = git.Repo(self._worktree_root)
        except git.exc.InvalidGitRepositoryError:
            self._emit(ControllerEventType.LOG, {"text": "Gitリポジトリが存在しません。`git init` を実行してください。"})
            return False

        if not repo.is_dirty(untracked_files=True):
            if not quiet_when_no_change:
                self._emit(ControllerEventType.LOG, {"text": "コミット対象の変更がありません。"})
            return False

        try:
            repo.git.add(A=True)
        except Exception as exc:  # noqa: BLE001
            self._emit(ControllerEventType.LOG, {"text": f"git add に失敗しました: {exc}"})
            return False

        prefix = "sibyl-auto-save" if auto else "sibyl-manual-save"
        message = f"{prefix} {datetime.utcnow().isoformat(timespec='seconds')}Z"
        try:
            repo.index.commit(message)
        except Exception as exc:  # noqa: BLE001
            self._emit(ControllerEventType.LOG, {"text": f"コミットに失敗しました: {exc}"})
            return False

        self._emit(ControllerEventType.LOG, {"text": f"変更をコミットしました: {message}"})
        return True

    def _maybe_auto_commit(self) -> None:
        if not getattr(self, "_auto_commit_enabled", False):
            return
        self._perform_commit(auto=True, quiet_when_no_change=True)

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

    def _select_candidates(
        self,
        candidates: List[CandidateInfo],
        scoreboard: Optional[Dict[str, Dict[str, object]]] = None,
    ) -> SelectionDecision:
        candidates, scoreboard = self._prune_boss_candidates(candidates, scoreboard)

        mode = getattr(self, "_flow_mode", FlowMode.MANUAL)
        if mode in {FlowMode.MANUAL, FlowMode.AUTO_REVIEW}:
            future = self._request_selection(candidates, scoreboard)
            return future.result()

        decision = self._auto_select_decision(candidates, scoreboard or {})
        if decision is None:
            future = self._request_selection(candidates, scoreboard)
            return future.result()

        candidate_map = {candidate.key: candidate for candidate in candidates}
        selected_candidate = candidate_map.get(decision.selected_key)
        label = selected_candidate.label if selected_candidate else decision.selected_key
        self._emit(
            ControllerEventType.LOG,
            {
                "text": f"[flow {mode.value}] {label} を自動選択しました。",
            },
        )
        return decision

    def _auto_select_decision(
        self,
        candidates: List[CandidateInfo],
        scoreboard: Dict[str, Dict[str, object]],
    ) -> Optional[SelectionDecision]:
        boss_mode = self._config.boss_mode
        if boss_mode == BossMode.SKIP:
            return None

        candidate_map = {candidate.key: candidate for candidate in candidates}

        def score_from_entry(entry: Dict[str, object]) -> Optional[float]:
            score = entry.get("score")
            if score is None:
                return None
            try:
                return float(score)
            except (TypeError, ValueError):
                return None

        selected_key: Optional[str]
        if boss_mode == BossMode.REWRITE and "boss" in candidate_map:
            selected_key = "boss"
        else:
            selected_key = None
            best_score: Optional[float] = None
            for candidate in candidates:
                entry = scoreboard.get(candidate.key, {})
                score_value = score_from_entry(entry)
                if score_value is None:
                    continue
                if best_score is None or score_value > best_score:
                    best_score = score_value
                    selected_key = candidate.key
            if selected_key is None:
                return None

        scores: Dict[str, float] = {}
        comments: Dict[str, str] = {}
        for candidate in candidates:
            entry = scoreboard.get(candidate.key, {})
            score_value = score_from_entry(entry)
            scores[candidate.key] = score_value if score_value is not None else 0.0
            comment_value = entry.get("comment")
            if isinstance(comment_value, str) and comment_value:
                comments[candidate.key] = comment_value

        return SelectionDecision(selected_key=selected_key, scores=scores, comments=comments)

    def _prune_boss_candidates(
        self,
        candidates: List[CandidateInfo],
        scoreboard: Optional[Dict[str, Dict[str, object]]],
    ) -> tuple[List[CandidateInfo], Optional[Dict[str, Dict[str, object]]]]:
        if self._config.boss_mode == BossMode.REWRITE:
            return candidates, scoreboard
        filtered = [candidate for candidate in candidates if candidate.key != "boss"]
        if not scoreboard or "boss" not in scoreboard:
            return filtered, scoreboard
        trimmed = {key: value for key, value in scoreboard.items() if key != "boss"}
        return filtered, trimmed

    def history_previous(self) -> Optional[str]:
        return self._history.history_previous()

    def history_next(self) -> Optional[str]:
        return self._history.history_next()

    def history_reset(self) -> None:
        self._history.reset_cursor()

    @property
    def _cycle_history(self) -> List[Dict[str, object]]:
        return self._history._cycle_history

    @_cycle_history.setter
    def _cycle_history(self, value: List[Dict[str, object]]) -> None:
        self._history.set_cycle_history(value)

    def _perform_revert(self, silent: bool = False) -> None:
        tmux_manager = self._last_tmux_manager
        pane_ids = self._tmux_list_panes() or []
        main_pane = pane_ids[0] if pane_ids else None

        snapshot = self._history.last_snapshot()
        if snapshot is None:
            session_id: Optional[str]
            if self._pre_cycle_selected_session_set and self._pre_cycle_selected_session:
                session_id = self._pre_cycle_selected_session
            elif self._active_main_session_id:
                session_id = self._active_main_session_id
            else:
                session_id = self._last_started_main_session_id
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
                self._emit(ControllerEventType.LOG, {"text": f"前回のセッションを再開しました。次の指示はセッション {summary} から再開します。"})
                self._emit_status("待機中")
                self._emit_pause_state()
            self._pre_cycle_selected_session = None
            self._pre_cycle_selected_session_set = False
            return

        snapshot = self._history.last_snapshot() or {}
        session_id = snapshot.get("selected_session") or self._active_main_session_id or self._last_started_main_session_id

        self._last_selected_session = session_id
        self._active_main_session_id = session_id
        self._last_scoreboard = snapshot.get("scoreboard", {})
        self._last_instruction = snapshot.get("instruction")
        if self._last_scoreboard:
            self._emit(ControllerEventType.SCOREBOARD, {"scoreboard": self._last_scoreboard})

        self._paused = False
        if tmux_manager and main_pane:
            if session_id:
                tmux_manager.promote_to_main(session_id=session_id, pane_id=main_pane)
            else:
                tmux_manager.launch_main_session(pane_id=main_pane)

        summary = session_id or "(未選択)"
        if not silent:
            self._emit(ControllerEventType.LOG, {"text": f"サイクルを巻き戻しました。次の指示はセッション {summary} から再開します。"})
            self._emit_status("待機中")
            self._emit_pause_state()
        self._pre_cycle_selected_session = None
        self._pre_cycle_selected_session_set = False

    def _emit_pause_state(self) -> None:
        self._emit(ControllerEventType.PAUSE_STATE, {"paused": self._paused})

    def _on_main_session_started(self, session_id: str) -> None:
        self._active_main_session_id = session_id
        self._last_started_main_session_id = session_id
        if self._last_selected_session is None:
            self._last_selected_session = session_id
        self._config.reuse_existing_session = True

    async def _run_instruction(self, instruction: str) -> None:
        await self._workflow.run(instruction)

    def _resolve_selection(self, index: int) -> None:
        if not self._selection_context:
            self._emit(ControllerEventType.LOG, {"text": "現在選択待ちではありません。"})
            return
        context = self._selection_context
        if index < 1 or index > len(context.candidates):
            self._emit(ControllerEventType.LOG, {"text": "無効な番号です。"})
            return
        candidate = context.candidates[index - 1]
        scores = {
            cand.key: (1.0 if cand.key == candidate.key else 0.0) for cand in context.candidates
        }
        decision = SelectionDecision(selected_key=candidate.key, scores=scores)
        context.future.set_result(decision)
        self._emit(ControllerEventType.LOG, {"text": f"{candidate.label} を選択しました。"})
        self._selection_context = None
        self._emit(ControllerEventType.SELECTION_FINISHED, {})

    async def _handle_attach_command(self, *, force: bool = False) -> None:
        session_name = self._config.tmux_session
        wait_for_session = not force and self._attach_mode == "auto"
        if wait_for_session:
            self._emit(ControllerEventType.LOG, {"text": f"[auto] tmuxセッション {session_name} の起動を待機中..."})
            session_ready = await self._wait_for_session(session_name)
            if not session_ready:
                self._emit(
                    ControllerEventType.LOG,
                    {"text": f"[auto] tmuxセッション {session_name} が見つかりませんでした。少し待ってから再度試してください。"},
                )
                return
        else:
            if not self._attach_manager.session_exists(session_name):
                self._emit(
                    ControllerEventType.LOG,
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
                    ControllerEventType.LOG,
                    {"text": f"[auto] tmuxセッション {session_name} は既に接続済みのため、自動アタッチをスキップしました。"},
                )
                return
        result = self._attach_manager.attach(session_name, workdir=self._worktree_root)
        if result.returncode == 0:
            prefix = "[auto] " if perform_detection else ""
            self._emit(ControllerEventType.LOG, {"text": f"{prefix}tmuxセッション {session_name} に接続しました。"})
        else:
            self._emit(ControllerEventType.LOG, {"text": "tmuxへの接続に失敗しました。tmuxが利用可能か確認してください。"})

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
            self._emit(ControllerEventType.LOG, {"text": "保存済みセッションが見つかりません。"})
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
        self._emit(ControllerEventType.LOG, {"text": "\n".join(lines)})
        self._emit(ControllerEventType.LOG, {"text": "再開するには /resume からセッションを選択してください。"})

    def _load_session(self, index: int) -> None:
        if not self._resume_options:
            self._emit(ControllerEventType.LOG, {"text": "先に /resume で一覧を表示してください。"})
            return
        if index < 1 or index > len(self._resume_options):
            self._emit(ControllerEventType.LOG, {"text": "無効な番号です。"})
            return
        reference = self._resume_options[index - 1]
        try:
            manifest = self._manifest_store.load_manifest(reference.session_id)
        except FileNotFoundError:
            self._emit(ControllerEventType.LOG, {"text": "セッションファイルが見つかりませんでした。"})
            return
        self._apply_manifest(manifest)
        self._emit(ControllerEventType.LOG, {"text": f"セッション {manifest.session_id} を読み込みました。"})
        if manifest.scoreboard:
            self._emit(ControllerEventType.SCOREBOARD, {"scoreboard": manifest.scoreboard})
        self._show_conversation_log(manifest.conversation_log)

    def _apply_manifest(self, manifest: SessionManifest) -> None:
        self._config.session_id = manifest.session_id
        self._config.tmux_session = manifest.tmux_session
        self._config.worker_count = manifest.worker_count
        self._config.mode = SessionMode(manifest.mode)
        if manifest.logs_dir:
            self._config.logs_root = Path(manifest.logs_dir).parent
        else:
            self._config.logs_root = _ensure_logs_directory(manifest.session_id)
        self._config.reuse_existing_session = True
        self._last_scoreboard = manifest.scoreboard or {}
        self._last_instruction = manifest.latest_instruction
        self._last_selected_session = manifest.selected_session_id
        self._session_namespace = manifest.session_id
        self._emit_status("再開準備完了")
        self._ensure_tmux_session(manifest)

    def _ensure_tmux_session(self, manifest: SessionManifest) -> None:
        try:
            import libtmux
        except ImportError:
            self._emit(ControllerEventType.LOG, {"text": "libtmux が見つかりません。tmux セッションは手動で復元してください。"})
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
            session_namespace=manifest.session_id,
            boss_mode=self._config.boss_mode,
            project_root=self._worktree_root,
            worktree_storage_root=self._worktree_storage_root,
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
            self._emit(ControllerEventType.LOG, {"text": "会話ログはありません。"})
            return
        path = Path(log_path)
        if not path.exists():
            self._emit(ControllerEventType.LOG, {"text": "会話ログが見つかりませんでした。"})
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
        self._emit(ControllerEventType.LOG, {"text": "\n".join(lines)})

    def _emit_status(self, message: str) -> None:
        self._emit(ControllerEventType.STATUS, {"message": message})

    def _emit(self, event_type: Union[str, ControllerEventType], payload: Dict[str, object]) -> None:
        if isinstance(event_type, ControllerEventType):
            key = event_type.value
        else:
            key = event_type
        self._event_handler(key, payload)

    def _flow_mode_display(self, mode: Optional[FlowMode] = None) -> str:
        current = mode or getattr(self, "_flow_mode", FlowMode.MANUAL)
        if isinstance(current, FlowMode):
            return FLOW_MODE_LABELS.get(current, current.value)
        return str(current)

    def _merge_strategy_display(self, strategy: Optional[MergeMode] = None) -> str:
        current = strategy or getattr(self, "_merge_mode", MergeMode.MANUAL)
        if isinstance(current, MergeMode):
            return MERGE_STRATEGY_LABELS.get(current, current.value)
        return str(current)

    def _create_initial_config(self) -> SessionConfig:
        session_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + datetime.utcnow().strftime("%f")[:6]
        tmux_session = f"parallel-dev-{session_id}"
        logs_root = _ensure_logs_directory(session_id)
        return SessionConfig(
            session_id=session_id,
            tmux_session=tmux_session,
            worker_count=3,
            mode=SessionMode.PARALLEL,
            logs_root=logs_root,
            boss_mode=BossMode.SCORE,
            flow_mode=FlowMode.FULL_AUTO,
            merge_mode=MergeMode.AUTO,
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

    def _handle_merge_outcome(self, outcome: Optional[MergeOutcome]) -> None:
        if outcome is None:
            return
        branch = outcome.branch or "選択されたブランチ"
        status = outcome.status
        error = outcome.error
        reason_key = outcome.reason
        reason_labels = {
            "agent_auto": "エージェントが統合を担当",
        }
        if status == "delegate" and reason_key == "manual_user":
            self._emit(
                ControllerEventType.LOG,
                {
                    "text": (
                        f"[merge] manualモード: {branch} の統合作業はユーザに委譲されています。"
                        "必要な作業を終えたら /done で次に進んでください。"
                    )
                },
            )
            self._emit_status("統合作業待ち")
        elif status == "delegate" and reason_key == "agent_auto":
            self._emit(
                ControllerEventType.LOG,
                {
                    "text": (
                        f"[merge] Autoモード: {branch} の統合作業はエージェントが完了し、"
                        "ホストは結果を同期済みです。問題があればログを確認してください。"
                    )
                },
            )
            self._emit_status("統合作業完了")
        elif status == "delegate":
            label = reason_labels.get(reason_key, reason_key or "手動統合に切り替え")
            detail = f" 詳細: {error}" if error else ""
            self._emit(
                ControllerEventType.LOG,
                {"text": f"[merge] 自動マージをスキップし、エージェントに委譲します ({label}).{detail}"},
            )
            self._emit_status("統合作業待ち")
        elif status == "failed":
            self._emit(
                ControllerEventType.LOG,
                {"text": f"[merge] {branch} の自動マージに失敗しました: {error or '理由不明'}. /merge コマンドで戦略を切り替えてください。"},
            )
            self._emit_status("マージ失敗")

    def _handle_orchestrator_log(self, message: str) -> None:
        status = None
        token = "::status::"
        if token in message:
            message, status = message.split(token, 1)
            message = message.rstrip()
            status = status.strip()
        self._emit(ControllerEventType.LOG, {"text": message})
        if status:
            self._emit_status(status)

    @staticmethod
    def _default_builder(
        *,
        worker_count: int,
        log_dir: Optional[Path],
        session_name: Optional[str] = None,
        reuse_existing_session: bool = False,
        session_namespace: Optional[str] = None,
        boss_mode: BossMode = BossMode.SCORE,
        project_root: Optional[Path] = None,
        worktree_storage_root: Optional[Path] = None,
    ) -> Orchestrator:
        raise RuntimeError("Orchestrator builder is not configured.")


def build_orchestrator(
    *,
    worker_count: int,
    log_dir: Optional[Path],
    session_name: Optional[str] = None,
    reuse_existing_session: bool = False,
    session_namespace: Optional[str] = None,
    boss_mode: BossMode = BossMode.SCORE,
    project_root: Optional[Path] = None,
    worktree_storage_root: Optional[Path] = None,
    log_hook: Optional[Callable[[str], None]] = None,
    merge_mode: MergeMode = MergeMode.MANUAL,
) -> Orchestrator:
    session_name = session_name or "parallel-dev"
    timestamp = datetime.utcnow().strftime("%y-%m-%d-%H%M%S")
    if log_dir:
        base_logs_dir = Path(log_dir)
        base_logs_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_logs_dir = _ensure_logs_directory(timestamp)
    map_session_id = session_namespace or session_name or "parallel-dev"
    session_map_dir = default_config_dir() / "session_maps"
    session_map_dir.mkdir(parents=True, exist_ok=True)
    session_map_path = session_map_dir / f"{map_session_id}.yaml"

    project_root_path = Path(project_root).expanduser() if project_root else Path.cwd()
    storage_root_path = (
        Path(worktree_storage_root).expanduser()
        if worktree_storage_root
        else project_root_path
    )
    session_root = storage_root_path / ".parallel-dev"
    if session_namespace:
        session_root = session_root / "sessions" / session_namespace
    session_root.mkdir(parents=True, exist_ok=True)

    monitor = CodexMonitor(
        logs_dir=base_logs_dir,
        session_map_path=session_map_path,
        session_namespace=session_namespace,
    )
    tmux_manager = TmuxLayoutManager(
        session_name=session_name,
        worker_count=worker_count,
        monitor=monitor,
        root_path=project_root_path,
        startup_delay=0.0,
        backtrack_delay=0.0,
        reuse_existing_session=reuse_existing_session,
        session_namespace=session_namespace,
    )
    worktree_manager = WorktreeManager(
        root=project_root_path,
        worker_count=worker_count,
        session_namespace=session_namespace,
        storage_root=storage_root_path,
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
        log_hook=log_hook,
        merge_mode=merge_mode,
    )
