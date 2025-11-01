"""Typer CLI entrypoint for the parallel developer orchestrator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import typer

from .orchestrator import CandidateInfo, Orchestrator, SelectionDecision
from .services import (
    BossManager,
    CodexMonitor,
    LogManager,
    TmuxLayoutManager,
    WorktreeManager,
)

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
    decision_selector = build_interactive_selector()
    result = orchestrator.run_cycle(instruction, selector=decision_selector)

    typer.echo("\n=== Final Scoreboard ===")
    for key, data in sorted(
        result.sessions_summary.items(),
        key=lambda item: item[1].get("score", 0.0),
        reverse=True,
    ):
        score = data.get("score", 0.0)
        comment = data.get("comment", "")
        typer.echo(f"{key:>10}: {score:.2f}" + (f"  # {comment}" if comment else ""))

    typer.echo(f"\n[parallel-dev] Selected session: {result.selected_session}")


def build_orchestrator(worker_count: int, log_dir: Optional[Path]) -> Orchestrator:
    """Factory for an Orchestrator instance."""

    session_name = "parallel-dev"
    timestamp = datetime.utcnow().strftime("%y-%m-%d-%H%M%S")

    if log_dir is None:
        base_logs_dir = Path("logs") / timestamp
    else:
        base_logs_dir = Path(log_dir)

    base_logs_dir.mkdir(parents=True, exist_ok=True)
    session_map_path = base_logs_dir / "sessions_map.yaml"

    monitor = CodexMonitor(
        logs_dir=base_logs_dir,
        session_map_path=session_map_path,
    )
    tmux_manager = TmuxLayoutManager(
        session_name=session_name,
        worker_count=worker_count,
        monitor=monitor,
        root_path=Path.cwd(),
    )
    worktree_manager = WorktreeManager(root=Path.cwd(), worker_count=worker_count)
    boss_manager = BossManager()
    log_manager = LogManager(logs_dir=base_logs_dir)

    return Orchestrator(
        tmux_manager=tmux_manager,
        worktree_manager=worktree_manager,
        monitor=monitor,
        boss_manager=boss_manager,
        log_manager=log_manager,
        worker_count=worker_count,
        session_name=session_name,
    )


def build_interactive_selector():
    """Create a selector callable that prompts the user for scores and a choice."""

    def selector(candidates: List[CandidateInfo]) -> SelectionDecision:
        typer.echo("=== Candidate Evaluation ===")
        scores: Dict[str, float] = {}
        comments: Dict[str, str] = {}

        for candidate in candidates:
            typer.echo(
                f"[{candidate.key}] {candidate.label} | branch={candidate.branch} | worktree={candidate.worktree}"
            )
            score = typer.prompt(
                f"Score for {candidate.key}",
                type=float,
                default=0.0,
            )
            comment = typer.prompt(
                f"Comment for {candidate.key} (optional)",
                default="",
            )
            scores[candidate.key] = score
            if comment.strip():
                comments[candidate.key] = comment.strip()

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        default_choice = sorted_keys[0] if sorted_keys else candidates[0].key
        available = ", ".join(sorted(scores.keys()))
        selection = typer.prompt(
            f"Select candidate to adopt ({available})",
            default=default_choice,
        )
        while selection not in scores:
            selection = typer.prompt(
                f"Invalid choice. Select one of ({available})",
                default=default_choice,
            )

        typer.echo("\n=== Provisional Scoreboard ===")
        for key in sorted_keys:
            summary = f"{scores[key]:.2f}"
            if key == selection:
                summary += "  <-- selected"
            typer.echo(f"{key:>10}: {summary}")

        return SelectionDecision(
            selected_key=selection,
            scores=scores,
            comments=comments,
        )

    return selector
