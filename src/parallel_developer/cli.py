"""Textual-based interactive CLI for parallel developer orchestrator."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional
import platform
import subprocess
import shlex
import shutil
from subprocess import PIPE

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Footer, Header, Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from .orchestrator import CandidateInfo, OrchestrationResult, Orchestrator, SelectionDecision
from .session_manifest import ManifestStore, PaneRecord, SessionManifest, SessionReference
from .services import CodexMonitor, LogManager, TmuxLayoutManager, WorktreeManager


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

    @property
    def run_parallel(self) -> bool:
        return self.mode == SessionMode.PARALLEL and self.worker_count > 0


@dataclass
class SelectionContext:
    future: Future
    candidates: List[CandidateInfo]
    scoreboard: Dict[str, Dict[str, object]]


class ControllerEvent(Message):
    def __init__(self, event_type: str, payload: Optional[Dict[str, object]] = None) -> None:
        super().__init__()
        self.event_type = event_type
        self.payload = payload or {}


class StatusPanel(Static):
    def update_status(self, config: SessionConfig, message: str) -> None:
        lines = [
            f"tmux session : {config.tmux_session}",
            f"mode         : {config.mode.value}",
            f"workers      : {config.worker_count}",
            f"logs root    : {config.logs_root}",
            f"status       : {message}",
        ]
        self.update("\n".join(lines))


class EventLog(RichLog):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, highlight=True, markup=True, **kwargs)

    def log(self, text: str) -> None:
        for line in text.splitlines():
            self.write(line)


class CommandHint(Static):
    def update_hint(self) -> None:
        self.update(
            "Commands : /parallel <n>, /mode main|parallel, /resume, /load <n>, /status, /scoreboard, /pick <n>, /help, /exit"
        )


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
        self._attach_manager = TmuxAttachManager()
        self._settings_path = Path(settings_path) if settings_path else (self._worktree_root / ".parallel-dev" / "settings.json")
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        self._settings: Dict[str, object] = self._load_settings()
        self._attach_mode: str = str(self._settings.get("attach_mode", "auto"))

    async def handle_input(self, user_input: str) -> None:
        text = user_input.strip()
        if not text:
            return
        if text.startswith("/"):
            await self._handle_command(text)
        else:
            await self._run_instruction(text)

    async def _handle_command(self, command: str) -> None:
        parts = command.split()
        name = parts[0].lower()

        if name in {"/exit", "/quit"}:
            self._emit("quit", {})
            return

        if name == "/help":
            self._emit(
                "log",
                {
                    "text": "利用可能なコマンド:\n"
                    "  /parallel <n>   : ワーカー数を n に設定\n"
                    "  /mode main      : メインのみで実行\n"
                    "  /mode parallel  : 並列実行に戻す\n"
                    "  /resume         : 保存済みセッションを一覧\n"
                    "  /load <n>       : 一覧から指定番号を再開\n"
                    "  /status         : 現在の状態を表示\n"
                    "  /scoreboard     : 直近スコアを表示\n"
                    "  /pick <n>       : 候補選択\n"
                    "  /help           : このヘルプ\n"
                    "  /exit           : 終了"
                },
            )
            return

        if name == "/status":
            self._emit_status("待機中")
            return

        if name == "/parallel":
            if len(parts) != 2 or not parts[1].isdigit():
                self._emit("log", {"text": "使い方: /parallel <ワーカー数>"})
                return
            self._config.worker_count = int(parts[1])
            self._emit_status("設定を更新しました。")
            return

        if name == "/mode":
            if len(parts) != 2 or parts[1].lower() not in {"main", "parallel"}:
                self._emit("log", {"text": "使い方: /mode main | /mode parallel"})
                return
            self._config.mode = SessionMode(parts[1].lower())
            self._emit_status("設定を更新しました。")
            return

        if name == "/attach":
            if len(parts) == 2:
                mode = parts[1].lower()
                if mode in {"auto", "manual"}:
                    self._attach_mode = mode
                    self._emit("log", {"text": f"/attach モードを {mode} に設定しました。"})
                    self._save_settings()
                    return
                self._emit("log", {"text": "使い方: /attach [auto|manual]"})
                return
            await self._handle_attach_command(force=True)
            return

        if name == "/scoreboard":
            self._emit("scoreboard", {"scoreboard": self._last_scoreboard})
            return

        if name == "/resume":
            self._list_sessions()
            return

        if name == "/load":
            if len(parts) != 2 or not parts[1].isdigit():
                self._emit("log", {"text": "使い方: /load <番号>"})
                return
            index = int(parts[1])
            self._load_session(index)
            return

        if name == "/pick":
            if len(parts) != 2 or not parts[1].isdigit():
                self._emit("log", {"text": "使い方: /pick <番号>"})
                return
            self._resolve_selection(int(parts[1]))
            return

        self._emit("log", {"text": f"未知のコマンドです: {command}"})

    async def _run_instruction(self, instruction: str) -> None:
        if self._running:
            self._emit("log", {"text": "別の指示を処理中です。完了を待ってから再度実行してください。"})
            return
        if self._selection_context:
            self._emit("log", {"text": "候補選択待ちです。/pick <n> で選択してください。"})
            return

        self._running = True
        self._emit_status("メインセッションを準備中...")

        logs_dir = self._create_cycle_logs_dir()

        orchestrator = self._builder(
            worker_count=self._config.worker_count,
            log_dir=logs_dir,
            session_name=self._config.tmux_session,
            reuse_existing_session=self._config.reuse_existing_session,
        )

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

        auto_attach_task = self._schedule_auto_attach()
        try:
            self._emit("log", {"text": f"指示を開始: {instruction}"})
            result: OrchestrationResult = await loop.run_in_executor(None, run_cycle)
            self._last_scoreboard = dict(result.sessions_summary)
            self._last_instruction = instruction
            self._last_selected_session = result.selected_session
            self._config.reuse_existing_session = True
            self._emit("scoreboard", {"scoreboard": self._last_scoreboard})
            self._emit("log", {"text": "指示が完了しました。"})
            if result.artifact:
                manifest = self._build_manifest(result, logs_dir)
                self._manifest_store.save_manifest(manifest)
                self._emit("log", {"text": f"セッションを保存しました: {manifest.session_id}"})
        except Exception as exc:  # pylint: disable=broad-except
            self._emit("log", {"text": f"エラーが発生しました: {exc}"})
        finally:
            self._selection_context = None
            self._running = False
            self._emit_status("待機中")
            await self._await_auto_attach(auto_attach_task)

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
        self._emit("log", {"text": "再開するには /load <番号> を入力してください。"})

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

    def _schedule_auto_attach(self) -> Optional[asyncio.Task[None]]:
        if self._attach_mode != "auto":
            return None
        loop = asyncio.get_running_loop()
        return loop.create_task(
            self._handle_attach_command(force=False),
            name="parallel-dev-auto-attach",
        )

    async def _await_auto_attach(self, task: Optional[asyncio.Task[None]]) -> None:
        if not task:
            return
        try:
            await task
        except Exception:  # pragma: no cover - logging handled inside run
            self._emit("log", {"text": "[auto] tmuxへの接続処理でエラーが発生しました。"})

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
        border: round $accent;
        margin-bottom: 1;
    }

    #status {
        border: round $accent-lighten-1;
        padding: 1;
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
    }
    """

    BINDINGS = [
        ("ctrl+c", "quit", "終了"),
        ("ctrl+q", "quit", "終了"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.status_panel: Optional[StatusPanel] = None
        self.log_panel: Optional[EventLog] = None
        self.selection_list: Optional[OptionList] = None
        self.command_input: Optional[Input] = None
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
                hint = CommandHint(id="hint")
                hint.update_hint()
                yield hint
                self.command_input = Input(placeholder="指示または /コマンド", id="command")
                yield self.command_input
        yield Footer()

    async def on_mount(self) -> None:
        if self.command_input:
            self.command_input.focus()
        self._post_event("status", {"message": "待機中"})

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self.command_input:
            self.command_input.value = ""
        asyncio.create_task(self.controller.handle_input(event.value))

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

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        try:
            index = int(event.option_id)
        except (TypeError, ValueError):
            return
        await self.controller.handle_input(f"/pick {index}")


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
    session_map_path = base_logs_dir / "sessions_map.yaml"

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
