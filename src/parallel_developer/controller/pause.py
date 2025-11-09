"""Pause/escape handling helpers for CLIController."""

from __future__ import annotations

import asyncio
import subprocess
from subprocess import PIPE
from typing import List, Optional, TYPE_CHECKING

from .events import ControllerEventType

if TYPE_CHECKING:  # pragma: no cover
    from . import CLIController


class PauseHelper:
    """Encapsulates pause state transitions and broadcast logic."""

    def __init__(self, controller: "CLIController") -> None:
        self._controller = controller

    def handle_escape(self) -> None:
        controller = self._controller
        controller.broadcast_escape()
        if not controller._paused:
            controller._paused = True
            controller._emit(ControllerEventType.LOG, {"text": "一時停止モードに入りました。追加指示は現在のワーカーペインへ送信されます。"})
            controller._emit_status("一時停止モード")
            controller._emit_pause_state()
            return
        if controller._running:
            current_id = controller._current_cycle_id
            if current_id is not None:
                controller._cancelled_cycles.add(current_id)
            controller._current_cycle_id = None
            controller._running = False
        if controller._continue_future and not controller._continue_future.done():
            controller._continue_future.set_result("done")
        if controller._continuation_input_future and not controller._continuation_input_future.done():
            controller._continuation_input_future.set_result("")
        controller._awaiting_continuation_input = False
        controller._paused = False
        controller._emit(ControllerEventType.LOG, {"text": "現在のサイクルをキャンセルし、前の状態へ戻しました。"})
        controller._emit_status("待機中")
        controller._emit_pause_state()
        controller._perform_revert(silent=True)

    async def dispatch_paused_instruction(self, instruction: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._send_instruction_to_panes(instruction))

    def _send_instruction_to_panes(self, instruction: str) -> None:
        controller = self._controller
        session_name = controller._config.tmux_session
        pane_ids = self._tmux_list_panes()
        if pane_ids is None:
            return
        if len(pane_ids) <= 2:
            controller._emit(ControllerEventType.LOG, {"text": f"tmuxセッション {session_name} にワーカーペインが見つからず、追加指示を送信できませんでした。"})
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
        controller._emit(ControllerEventType.LOG, {"text": f"[pause] {len(worker_panes)} ワーカーペインへ追加指示を送信: {preview}"})
        controller._paused = False
        controller._emit_pause_state()
        controller._emit_status("待機中")

    def _tmux_list_panes(self) -> Optional[List[str]]:
        controller = self._controller
        session_name = controller._config.tmux_session
        try:
            result = subprocess.run(
                ["tmux", "list-panes", "-t", session_name, "-F", "#{pane_id}"],
                check=False,
                stdout=PIPE,
                stderr=PIPE,
                text=True,
            )
        except FileNotFoundError:
            controller._emit(ControllerEventType.LOG, {"text": "tmux コマンドが見つかりません。tmuxがインストールされているか確認してください。"})
            return None
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            if message:
                controller._emit(ControllerEventType.LOG, {"text": f"tmux list-panes に失敗しました: {message}"})
            return None
        return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
