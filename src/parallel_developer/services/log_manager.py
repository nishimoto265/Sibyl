"""オーケストレーションログを集約するサービス."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml


class LogManager:
    """各サイクルのメタデータを YAML/JSONL として永続化する."""

    def __init__(self, logs_dir: Path) -> None:
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.cycles_dir = self.logs_dir / "cycles"
        self.cycles_dir.mkdir(parents=True, exist_ok=True)

    def record_cycle(
        self,
        *,
        instruction: str,
        layout: Mapping[str, Any],
        fork_map: Mapping[str, str],
        completion: Mapping[str, Any],
        result: Any,
    ) -> Dict[str, Path]:
        timestamp = datetime.utcnow().strftime("%y-%m-%d-%H%M%S")
        payload = {
            "instruction": instruction,
            "layout": dict(layout),
            "fork_map": dict(fork_map),
            "completion": dict(completion),
            "result": {
                "selected_session": getattr(result, "selected_session", None),
                "sessions_summary": getattr(result, "sessions_summary", None),
            },
        }
        path = self.cycles_dir / f"{timestamp}.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        jsonl_path = self.cycles_dir / f"{timestamp}.jsonl"
        events = self._build_cycle_events(
            instruction=instruction,
            layout=layout,
            fork_map=fork_map,
            completion=completion,
            result=result,
        )
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        return {"yaml": path, "jsonl": jsonl_path}

    def _build_cycle_events(
        self,
        *,
        instruction: str,
        layout: Mapping[str, Any],
        fork_map: Mapping[str, str],
        completion: Mapping[str, Any],
        result: Any,
    ) -> List[Dict[str, Any]]:
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        events: List[Dict[str, Any]] = []

        events.append(
            {
                "type": "instruction",
                "timestamp": timestamp,
                "instruction": instruction,
                "layout": dict(layout),
            }
        )

        if fork_map:
            events.append(
                {
                    "type": "fork",
                    "timestamp": timestamp,
                    "fork_map": dict(fork_map),
                }
            )

        if completion:
            events.append(
                {
                    "type": "completion",
                    "timestamp": timestamp,
                    "completion": dict(completion),
                }
            )

        scoreboard = getattr(result, "sessions_summary", None) or {}
        if scoreboard:
            events.append(
                {
                    "type": "scoreboard",
                    "timestamp": timestamp,
                    "scoreboard": scoreboard,
                }
            )

        selected_session = getattr(result, "selected_session", None)
        selected_key = None
        for key, data in scoreboard.items():
            if data.get("selected"):
                selected_key = key
                break
        events.append(
            {
                "type": "selection",
                "timestamp": timestamp,
                "selected_session": selected_session,
                "selected_key": selected_key,
            }
        )

        artifact = getattr(result, "artifact", None)
        if artifact is not None:
            events.append(
                {
                    "type": "artifact",
                    "timestamp": timestamp,
                    "main_session_id": getattr(artifact, "main_session_id", None),
                    "worker_sessions": getattr(artifact, "worker_sessions", {}),
                    "boss_session_id": getattr(artifact, "boss_session_id", None),
                    "worker_paths": {k: str(v) for k, v in getattr(artifact, "worker_paths", {}).items()},
                    "boss_path": str(getattr(artifact, "boss_path", "")) if getattr(artifact, "boss_path", None) else None,
                }
            )

        return events
