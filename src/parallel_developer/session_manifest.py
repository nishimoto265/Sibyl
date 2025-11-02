"""Session manifest management for the interactive parallel developer CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


@dataclass(slots=True)
class PaneRecord:
    role: str  # "main" | "boss" | "worker"
    name: Optional[str]
    session_id: str
    worktree: Optional[str] = None


@dataclass(slots=True)
class SessionManifest:
    session_id: str
    created_at: str
    tmux_session: str
    worker_count: int
    mode: str
    logs_dir: str
    latest_instruction: Optional[str] = None
    scoreboard: Dict[str, Dict[str, object]] = field(default_factory=dict)
    conversation_log: Optional[str] = None
    main: PaneRecord = field(default_factory=lambda: PaneRecord(role="main", name=None, session_id=""))
    boss: Optional[PaneRecord] = None
    workers: Dict[str, PaneRecord] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "tmux_session": self.tmux_session,
            "worker_count": self.worker_count,
            "mode": self.mode,
            "logs_dir": self.logs_dir,
            "latest_instruction": self.latest_instruction,
            "scoreboard": self.scoreboard,
            "conversation_log": self.conversation_log,
            "main": asdict(self.main),
            "boss": asdict(self.boss) if self.boss else None,
            "workers": {name: asdict(record) for name, record in self.workers.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "SessionManifest":
        workers = {
            name: PaneRecord(**record)
            for name, record in (data.get("workers") or {}).items()
        }
        boss_data = data.get("boss")
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            tmux_session=data["tmux_session"],
            worker_count=int(data.get("worker_count", len(workers))),
            mode=data.get("mode", "parallel"),
            logs_dir=data["logs_dir"],
            latest_instruction=data.get("latest_instruction"),
            scoreboard=data.get("scoreboard", {}) or {},
            conversation_log=data.get("conversation_log"),
            main=PaneRecord(**data.get("main", {})),
            boss=PaneRecord(**boss_data) if boss_data else None,
            workers=workers,
        )


@dataclass(slots=True)
class SessionReference:
    session_id: str
    tmux_session: str
    manifest_path: Path
    created_at: str
    worker_count: int
    mode: str
    latest_instruction: Optional[str]
    logs_dir: Path


class ManifestStore:
    """Persist CLI session manifests and maintain an index for /resume."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or Path.home() / ".parallel-dev" / "manifests"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / "index.json"
        if not self.index_path.exists():
            self.index_path.write_text(json.dumps({"sessions": {}}), encoding="utf-8")

    def save_manifest(self, manifest: SessionManifest) -> None:
        manifest_path = self.base_dir / f"{manifest.session_id}.yaml"
        manifest_path.write_text(
            yaml.safe_dump(manifest.to_dict(), sort_keys=False),
            encoding="utf-8",
        )
        self._update_index(
            SessionReference(
                session_id=manifest.session_id,
                tmux_session=manifest.tmux_session,
                manifest_path=manifest_path,
                created_at=manifest.created_at,
                worker_count=manifest.worker_count,
                mode=manifest.mode,
                latest_instruction=manifest.latest_instruction,
                logs_dir=Path(manifest.logs_dir),
            )
        )

    def list_sessions(self) -> List[SessionReference]:
        data = self._load_index()
        sessions = []
        for session_id, payload in data.get("sessions", {}).items():
            try:
                sessions.append(
                    SessionReference(
                        session_id=session_id,
                        tmux_session=payload["tmux_session"],
                        manifest_path=Path(payload["manifest_path"]),
                        created_at=payload["created_at"],
                        worker_count=int(payload.get("worker_count", 0)),
                        mode=payload.get("mode", "parallel"),
                        latest_instruction=payload.get("latest_instruction"),
                        logs_dir=Path(payload["logs_dir"]),
                    )
                )
            except KeyError:
                continue
        sessions.sort(key=lambda ref: ref.created_at, reverse=True)
        return sessions

    def load_manifest(self, session_id: str) -> SessionManifest:
        data = self._load_index()
        payload = data.get("sessions", {}).get(session_id)
        if not payload:
            raise KeyError(f"Session {session_id!r} not found in manifest index.")
        manifest_path = Path(payload["manifest_path"])
        manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        return SessionManifest.from_dict(manifest_data)

    def _update_index(self, reference: SessionReference) -> None:
        data = self._load_index()
        data.setdefault("sessions", {})[reference.session_id] = {
            "tmux_session": reference.tmux_session,
            "manifest_path": str(reference.manifest_path),
            "created_at": reference.created_at,
            "worker_count": reference.worker_count,
            "mode": reference.mode,
            "latest_instruction": reference.latest_instruction,
            "logs_dir": str(reference.logs_dir),
        }
        self.index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_index(self) -> Dict[str, object]:
        return json.loads(self.index_path.read_text(encoding="utf-8"))
