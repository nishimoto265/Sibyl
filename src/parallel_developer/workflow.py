"""Workflow management for executing orchestrator cycles."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, TYPE_CHECKING

from .orchestrator import CandidateInfo, OrchestrationResult, SelectionDecision

if TYPE_CHECKING:
    from .controller import CLIController


class WorkflowManager:
    """Encapsulate the instruction execution flow for CLIController."""

    def __init__(self, controller: "CLIController") -> None:  # pragma: no cover - type check only
        self._controller = controller

    async def run_instruction(self, instruction: str) -> None:
        c = self._controller
        if c._running:
            c._emit("log", {"text": "別の指示を処理中です。完了を待ってから再度実行してください。"})
            return
        if c._selection_context:
            c._emit("log", {"text": "候補選択待ちです。/pick <n> で選択してください。"})
            return

        c._maybe_auto_commit()

        c._cycle_counter += 1
        cycle_id = c._cycle_counter
        c._current_cycle_id = cycle_id
        c._running = True
        c._emit_status("メインセッションを準備中...")
        c._active_main_session_id = None
        c._pre_cycle_selected_session = c._last_selected_session
        c._pre_cycle_selected_session_set = True

        logs_dir = c._create_cycle_logs_dir()

        orchestrator = c._builder(
            worker_count=c._config.worker_count,
            log_dir=logs_dir,
            session_name=c._config.tmux_session,
            reuse_existing_session=c._config.reuse_existing_session,
            session_namespace=c._session_namespace,
            boss_mode=c._config.boss_mode,
            project_root=c._worktree_root,
            worktree_storage_root=c._worktree_storage_root,
        )
        c._active_orchestrator = orchestrator
        c._last_tmux_manager = getattr(orchestrator, "_tmux", None)
        main_hook = getattr(orchestrator, "set_main_session_hook", None)
        if callable(main_hook):
            main_hook(c._on_main_session_started)
        worker_decider = getattr(orchestrator, "set_worker_decider", None)
        if callable(worker_decider):
            worker_decider(c._handle_worker_decision)

        loop = asyncio.get_running_loop()

        def selector(candidates: List[CandidateInfo], scoreboard: Optional[Dict[str, Dict[str, object]]] = None) -> SelectionDecision:
            return c._select_candidates(candidates, scoreboard)

        resume_session = c._last_selected_session

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
            c._emit("log", {"text": f"指示を開始: {instruction}"})
            if c._attach_mode == "auto":
                auto_attach_task = asyncio.create_task(c._handle_attach_command(force=False))
            result: OrchestrationResult = await loop.run_in_executor(None, run_cycle)
            continued = getattr(result, "continue_requested", False)
            if continued:
                c._last_selected_session = result.selected_session
                c._active_main_session_id = result.selected_session
                c._config.reuse_existing_session = True
                c._last_scoreboard = {}
                c._emit("log", {"text": "ワーカーを継続します。新しい指示を入力してください。"})
            elif cycle_id in c._cancelled_cycles:
                cancelled = True
                c._cancelled_cycles.discard(cycle_id)
            else:
                c._last_scoreboard = dict(result.sessions_summary)
                c._last_instruction = instruction
                c._last_selected_session = result.selected_session
                c._active_main_session_id = result.selected_session
                c._config.reuse_existing_session = True
                c._emit("scoreboard", {"scoreboard": c._last_scoreboard})
                c._emit("log", {"text": "指示が完了しました。"})
                if result.artifact:
                    manifest = c._build_manifest(result, logs_dir)
                    c._manifest_store.save_manifest(manifest)
                    c._emit("log", {"text": f"セッションを保存しました: {manifest.session_id}"})
                c._record_cycle_snapshot(result, cycle_id)
        except Exception as exc:  # pylint: disable=broad-except
            c._emit("log", {"text": f"エラーが発生しました: {exc}"})
        finally:
            c._selection_context = None
            if c._current_cycle_id == cycle_id:
                c._current_cycle_id = None
            c._running = False
            if cancelled:
                c._emit_status("待機中")
                c._emit_pause_state()
                c._perform_revert(silent=True)
            else:
                c._emit_status("一時停止中" if c._paused else "待機中")
                c._emit_pause_state()
            if auto_attach_task:
                try:
                    await auto_attach_task
                except Exception:  # noqa: BLE001
                    pass
            c._pre_cycle_selected_session = None
            c._pre_cycle_selected_session_set = False

        if continued:
            return

        if cancelled and c._queued_instruction:
            queued = c._queued_instruction
            c._queued_instruction = None
            await self.run_instruction(queued)
