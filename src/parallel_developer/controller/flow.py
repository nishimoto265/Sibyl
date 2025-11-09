"""Workerフロー制御の補助機能をまとめたモジュール。"""

from __future__ import annotations

from typing import Any, Mapping, Optional, TYPE_CHECKING

from .events import ControllerEventType
from ..orchestrator import CycleLayout, WorkerDecision

if TYPE_CHECKING:  # pragma: no cover
    from . import CLIController


class WorkerFlowHelper:
    """Controllerのワーカーフロー関連処理をまとめるヘルパー。"""

    def __init__(self, controller: "CLIController", flow_mode_cls) -> None:
        self._controller = controller
        self._controller_event_type = ControllerEventType
        self._flow_mode_cls = flow_mode_cls

    def handle_worker_decision(
        self,
        fork_map: Mapping[str, str],
        completion_info: Mapping[str, Any],
        layout: CycleLayout,
    ) -> WorkerDecision:
        c = self._controller
        flow_mode = getattr(c, "_flow_mode", self._flow_mode_cls.MANUAL)
        if flow_mode in {self._flow_mode_cls.AUTO_REVIEW, self._flow_mode_cls.FULL_AUTO}:
            c._emit(
                self._controller_event_type.LOG,
                {
                    "text": f"[flow {c._flow_mode_display()}] ワーカーの処理が完了しました。採点フェーズへ進みます。",
                },
            )
            return WorkerDecision(action="done")

        command = c._await_worker_command()
        if str(command).lower() == "continue":
            instruction = c._await_continuation_instruction()
            return WorkerDecision(action="continue", instruction=instruction)
        return WorkerDecision(action="done")
