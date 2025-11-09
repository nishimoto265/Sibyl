"""Controller → UI 間で利用するイベント種別を定義。"""

from __future__ import annotations

from enum import Enum


class ControllerEventType(str, Enum):
    STATUS = "status"
    LOG = "log"
    LOG_COPY = "log_copy"
    LOG_SAVE = "log_save"
    SCOREBOARD = "scoreboard"
    SELECTION_REQUEST = "selection_request"
    SELECTION_FINISHED = "selection_finished"
    PAUSE_STATE = "pause_state"
    QUIT = "quit"
