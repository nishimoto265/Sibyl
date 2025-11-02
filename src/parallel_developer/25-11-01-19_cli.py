"""Typer CLI entrypoint for the parallel developer orchestrator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, List, Optional

import typer

from .orchestrator import CandidateInfo, Orchestrator, SelectionDecision
from .services import CodexMonitor, LogManager, TmuxLayoutManager, WorktreeManager

app = typer.Typer(add_completion=False, no_args_is_help=True, invoke_without_command=True)


@app.callback()
def main(
    instruction: Annotated[
        str,
        typer.Option(
            "--instruction",
            "-i",
            prompt="指示を入力してください",
            help="Codexへ送信する指示文",
        ),
    ],
    workers: Annotated[
        int, typer.Option("--workers", "-w", min=1, help="並列workerの数")
    ] = 3,
    log_dir: Annotated[
        Optional[Path],
        typer.Option("--log-dir", help="ログ出力先ディレクトリ（未指定の場合は自動生成）"),
    ] = None,
) -> None:
    """Execute a full orchestrated cycle for the given instruction."""

    orchestrator = build_orchestrator(worker_count=workers, log_dir=log_dir)

    def selector(candidates: List[CandidateInfo], scoreboard: dict | None = None) -> SelectionDecision:
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
            choice = typer.prompt("採用する候補番号", type=int)
            if 1 <= choice <= len(candidates):
                break
            typer.echo("無効な番号です。もう一度入力してください。")

        selected = candidates[choice - 1]
        scores = {
            candidate.key: (1.0 if candidate.key == selected.key else 0.0)
            for candidate in candidates
        }
        return SelectionDecision(selected_key=selected.key, scores=scores)

    result = orchestrator.run_cycle(instruction, selector=selector)

    typer.echo("\n=== Scoreboard ===")
    def sort_key(item):
        score = item[1].get("score")
        return (score is None, -(score or 0.0))

    for key, data in sorted(result.sessions_summary.items(), key=sort_key):
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

    typer.echo(f"\n[parallel-dev] Selected session: {result.selected_session}")


def build_orchestrator(worker_count: int, log_dir: Optional[Path]) -> Orchestrator:
    """Factory for an Orchestrator instance."""

    session_name = "parallel-dev"
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
