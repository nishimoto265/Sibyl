"""Controller input/cycle history helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .controller import CLIController, Orchestrator, CycleLayout  # noqa: F401


class HistoryManager:
    """Handles input history and cycle snapshots for CLIController."""

    def __init__(self) -> None:
        self._input_history: List[str] = []
        self._history_cursor: int = 0
        self._cycle_history: List[Dict[str, object]] = []

    # Input history ---------------------------------------------------------
    def record_input(self, text: str) -> None:
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

    def reset_cursor(self) -> None:
        self._history_cursor = len(self._input_history)

    # Cycle history ---------------------------------------------------------
    def record_cycle_snapshot(self, result, cycle_id: int, last_instruction: Optional[str]) -> None:
        snapshot = {
            "cycle_id": cycle_id,
            "selected_session": result.selected_session,
            "scoreboard": dict(result.sessions_summary),
            "instruction": last_instruction,
        }
        self._cycle_history.append(snapshot)

    def last_snapshot(self) -> Optional[Dict[str, object]]:
        if not self._cycle_history:
            return None
        return self._cycle_history[-1]

    def pop_snapshot(self) -> Optional[Dict[str, object]]:
        if not self._cycle_history:
            return None
        return self._cycle_history.pop()

    def set_cycle_history(self, history_list: List[Dict[str, object]]) -> None:
        self._cycle_history = history_list
