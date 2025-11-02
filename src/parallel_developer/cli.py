"""Interactive CLI entrypoint for the parallel developer orchestrator."""

from __future__ import annotations

import textwrap
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import typer

from .orchestrator import CandidateInfo, Orchestrator, SelectionDecision
from .session_manifest import ManifestStore, PaneRecord, SessionManifest, SessionReference
from .services import CodexMonitor, LogManager, TmuxLayoutManager, WorktreeManager

app = typer.Typer(add_completion=False, invoke_without_command=True, no_args_is_help=False)


class SessionMode(str, Enum):
    PARALLEL = "parallel"
    MAIN = "main"


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


class InteractiveCLI:
    """Interactive shell to drive parallel developer orchestration cycles."""

    def __init__(
        self,
        *,
        orchestrator_builder: Callable[..., Orchestrator] = None,
        manifest_store: Optional[ManifestStore] = None,
        worktree_root: Optional[Path] = None,
    ) -> None:
        self._builder = orchestrator_builder or build_orchestrator
        self._manifest_store = manifest_store or ManifestStore()
        self._worktree_root = Path(worktree_root or Path.cwd())
        self._config = self._create_initial_config()
        self._last_scoreboard: Dict[str, Dict[str, object]] = {}
        self._last_instruction: Optional[str] = None

    def run(self) -> None:
        typer.echo("Parallel Developer CLI (type /help for commands, /exit to quit)")
        while True:
            try:
                raw = input("pdev> ").strip()
            except (EOFError, KeyboardInterrupt):
                typer.echo("\n終了します。")
                break

            if not raw:
                continue

            if raw.startswith("/"):
                if not self._handle_command(raw):
                    break
                continue

            self._handle_instruction(raw)

    # ------------------------------------------------------------------ #
    # Command handling
    # ------------------------------------------------------------------ #

    def _handle_command(self, command: str) -> bool:
        parts = command.split()
        name = parts[0].lower()

        if name in {"/exit", "/quit"}:
            typer.echo("セッションを終了します。")
            return False

        if name == "/help":
            self._print_help()
            return True

        if name == "/parallel":
            if len(parts) != 2 or not parts[1].isdigit():
                typer.echo("使い方: /parallel <ワーカー数>")
                return True
            new_count = int(parts[1])
            if new_count < 0:
                typer.echo("ワーカー数は0以上を指定してください。")
                return True
            self._config.worker_count = new_count
            typer.echo(f"ワーカー数を {new_count} に設定しました。")
            return True

        if name == "/mode":
            if len(parts) != 2 or parts[1].lower() not in {"main", "parallel"}:
                typer.echo("使い方: /mode main | /mode parallel")
                return True
            self._config.mode = SessionMode(parts[1].lower())
            typer.echo(f"モードを {self._config.mode.value} に切り替えました。")
            return True

        if name == "/scoreboard":
            if not self._last_scoreboard:
                typer.echo("スコアボード情報がありません。")
                return True
            self._print_scoreboard(self._last_scoreboard)
            return True

        if name == "/resume":
            self._resume_session()
            return True

        typer.echo(f"未知のコマンドです: {command}")
        return True

    def _print_help(self) -> None:
        typer.echo(
            textwrap.dedent(
                """
                利用可能なコマンド:
                  /parallel <n>   : ワーカー数を n に設定
                  /mode main      : メインのみで実行
                  /mode parallel  : 並列実行に戻す
                  /resume         : 過去セッションを再開
                  /scoreboard     : 直近のスコアボードを表示
                  /help           : このヘルプを表示
                  /exit           : CLIを終了
                """
            ).strip()
        )

    # ------------------------------------------------------------------ #
    # Instruction execution
    # ------------------------------------------------------------------ #

    def _handle_instruction(self, instruction: str) -> None:
        logs_dir = self._create_cycle_logs_dir()
        orchestrator = self._builder(
            worker_count=self._config.worker_count,
            log_dir=logs_dir,
            session_name=self._config.tmux_session,
            reuse_existing_session=self._config.reuse_existing_session,
        )

        def selector(candidates: List[CandidateInfo], scoreboard: dict | None = None) -> SelectionDecision:
            return self._selection_prompt(candidates, scoreboard)

        result = orchestrator.run_cycle(instruction, selector=selector)

        self._config.reuse_existing_session = True
        self._last_scoreboard = dict(result.sessions_summary)
        self._last_instruction = instruction

        self._print_scoreboard(self._last_scoreboard)

        if result.artifact:
            manifest = self._build_manifest(result, logs_dir)
            self._manifest_store.save_manifest(manifest)

    def _selection_prompt(
        self,
        candidates: List[CandidateInfo],
        scoreboard: Optional[Dict[str, Dict[str, object]]],
    ) -> SelectionDecision:
        typer.echo("\n=== Candidates ===")
        for idx, candidate in enumerate(candidates, start=1):
            typer.echo(f"{idx}. {candidate.label}")

        if scoreboard:
            typer.echo("\n--- Boss Scores ---")
            for key, data in sorted(
                scoreboard.items(),
                key=lambda item: (item[1].get("score") is None, -(item[1].get("score") or 0.0)),
            ):
                score = data.get("score")
                comment = data.get("comment", "")
                score_text = "-" if score is None else f"{score:.2f}"
                line = f"{key:>10}: {score_text}"
                if comment:
                    line += f"  # {comment}"
                typer.echo(line)

        while True:
            raw = input("採用する候補番号: ").strip()
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(candidates):
                    selected = candidates[idx - 1]
                    break
            typer.echo("無効な番号です。もう一度入力してください。")

        scores = {
            candidate.key: (1.0 if candidate.key == selected.key else 0.0)
            for candidate in candidates
        }
        return SelectionDecision(selected_key=selected.key, scores=scores)

    def _print_scoreboard(self, scoreboard: Dict[str, Dict[str, object]]) -> None:
        typer.echo("\n=== Scoreboard ===")

        def sort_key(item):
            score = item[1].get("score")
            return (score is None, -(score or 0.0))

        for key, data in sorted(scoreboard.items(), key=sort_key):
            score = data.get("score")
            comment = data.get("comment", "")
            selected = data.get("selected", False)
            score_text = "-" if score is None else f"{score:.2f}"
            line = f"{key:>10}: {score_text}"
            if selected:
                line += "  [selected]"
            if comment:
                line += f"  # {comment}"
            typer.echo(line)

    # ------------------------------------------------------------------ #
    # Resume logic
    # ------------------------------------------------------------------ #

    def _resume_session(self) -> None:
        references = self._manifest_store.list_sessions()
        if not references:
            typer.echo("再開できるセッションが見つかりません。")
            return

        typer.echo("\n=== 保存されたセッション ===")
        for idx, ref in enumerate(references, start=1):
            summary = ref.latest_instruction or ""
            typer.echo(
                f"{idx}. {ref.created_at} | tmux={ref.tmux_session} | workers={ref.worker_count} | mode={ref.mode}"
            )
            if summary:
                typer.echo(f"   last instruction: {summary[:80]}")

        while True:
            raw = input("再開するセッション番号 (キャンセルはEnter): ").strip()
            if not raw:
                typer.echo("再開をキャンセルしました。")
                return
            if raw.isdigit():
                idx = int(raw)
                if 1 <= idx <= len(references):
                    ref = references[idx - 1]
                    break
            typer.echo("無効な番号です。")

        manifest = self._manifest_store.load_manifest(ref.session_id)
        self._apply_manifest(manifest)
        typer.echo(f"セッション {manifest.session_id} を読み込みました。")
        self._show_conversation_log(manifest.conversation_log)

    def _apply_manifest(self, manifest: SessionManifest) -> None:
        self._config.session_id = manifest.session_id
        self._config.tmux_session = manifest.tmux_session
        self._config.worker_count = manifest.worker_count
        self._config.mode = SessionMode(manifest.mode)
        self._config.logs_root = Path(manifest.logs_dir).parent if manifest.logs_dir else Path("logs")
        self._config.reuse_existing_session = True

        self._last_scoreboard = dict(manifest.scoreboard or {})
        self._last_instruction = manifest.latest_instruction

        if not self._ensure_tmux_session(manifest):
            typer.echo("tmuxセッションを再作成できませんでした。")

    def _ensure_tmux_session(self, manifest: SessionManifest) -> bool:
        # Check if tmux session already exists
        import libtmux

        server = libtmux.Server()
        existing = server.find_where({"session_name": manifest.tmux_session})
        if existing is not None:
            return True

        worker_count = len(manifest.workers)
        orchestrator = self._builder(
            worker_count=worker_count,
            log_dir=Path(manifest.logs_dir) if manifest.logs_dir else None,
            session_name=manifest.tmux_session,
            reuse_existing_session=False,
        )

        tmux_manager = orchestrator._tmux  # type: ignore[attr-defined]
        orchestrator._worktree.prepare()  # type: ignore[attr-defined]
        tmux_manager.set_boss_path(Path(manifest.boss.worktree) if manifest.boss and manifest.boss.worktree else self._worktree_root)
        tmux_manager.set_reuse_existing_session(True)

        layout = tmux_manager.ensure_layout(
            session_name=manifest.tmux_session,
            worker_count=worker_count,
        )

        tmux_manager.resume_session(
            pane_id=layout.main_pane,
            workdir=self._worktree_root,
            session_id=manifest.main.session_id,
        )

        for index, worker_name in enumerate(layout.worker_names):
            record = manifest.workers.get(worker_name)
            if not record or not record.worktree:
                continue
            pane_id = layout.worker_panes[index]
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

        return True

    def _show_conversation_log(self, log_path: Optional[str]) -> None:
        if not log_path:
            return
        path = Path(log_path)
        if not path.exists():
            typer.echo("会話ログは利用できません。")
            return
        typer.echo("\n--- Conversation Log ---")
        typer.echo(path.read_text(encoding="utf-8"))
        typer.echo("--- End Conversation Log ---\n")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _create_initial_config(self) -> SessionConfig:
        session_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
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

        return SessionManifest(
            session_id=self._config.session_id,
            created_at=datetime.utcnow().isoformat(timespec="seconds"),
            tmux_session=self._config.tmux_session,
            worker_count=len(workers),
            mode=self._config.mode.value,
            logs_dir=str(logs_dir),
            latest_instruction=self._last_instruction,
            scoreboard=self._last_scoreboard,
            conversation_log=str(logs_dir / "instruction.log"),
            main=main_record,
            boss=boss_record,
            workers=workers,
        )


@app.callback(invoke_without_command=True)
def main() -> None:
    """Launch the interactive parallel developer CLI."""
    InteractiveCLI().run()


def build_orchestrator(
    worker_count: int,
    log_dir: Optional[Path],
    *,
    session_name: Optional[str] = None,
    reuse_existing_session: bool = False,
) -> Orchestrator:
    """Factory for an Orchestrator instance."""

    session_name = session_name or "parallel-dev"
    timestamp = datetime.utcnow().strftime("%y-%m-%d-%H%M%S")

    base_logs_dir = Path(log_dir) if log_dir else Path("logs") / timestamp
    base_logs_dir.mkdir(parents=True, exist_ok=True)
    session_map_path = base_logs_dir / "sessions_map.yaml"

    monitor = CodexMonitor(
        logs_dir=base_logs_dir,
        session_map_path=session_map_path,
    )
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
