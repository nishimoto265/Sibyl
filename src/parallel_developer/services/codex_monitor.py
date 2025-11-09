"""Codex セッションのロールアウトを監視するサービス."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Union

import yaml


class SessionReservationError(RuntimeError):
    """Codex rollout が既に別 namespace によって予約されている場合の例外."""

    def __init__(self, session_id: str, owner_namespace: Optional[str]) -> None:
        self.session_id = session_id
        self.owner_namespace = owner_namespace or "unknown"
        super().__init__(
            f"Codex session {session_id} is currently reserved by namespace '{self.owner_namespace}'. "
            "別の parallel-dev インスタンスが使用中のため、処理を中断します。"
        )


class CodexMonitor:
    """Codex rollout JSONL を読み取り、セッション状態／完了を追跡する."""

    def __init__(
        self,
        logs_dir: Path,
        session_map_path: Path,
        *,
        codex_sessions_root: Optional[Path] = None,
        poll_interval: float = 1.0,
        session_namespace: Optional[str] = None,
    ) -> None:
        self.logs_dir = Path(logs_dir)
        self.session_map_path = Path(session_map_path)
        self.poll_interval = poll_interval
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        (self.logs_dir / "sessions").mkdir(parents=True, exist_ok=True)
        if not self.session_map_path.exists():
            self.session_map_path.write_text("{}\n", encoding="utf-8")
        self.codex_sessions_root = (
            Path(codex_sessions_root)
            if codex_sessions_root is not None
            else Path.home() / ".codex" / "sessions"
        )
        self._session_namespace = session_namespace or "default"
        self._registry_dir = self.session_map_path.parent / "codex_session_registry"
        self._owned_sessions: Set[str] = set()
        self._forced_done: Set[str] = set()
        self._active_signal_paths: Dict[str, Path] = {}

    def register_session(self, *, pane_id: str, session_id: str, rollout_path: Path) -> None:
        try:
            offset = rollout_path.stat().st_size
        except OSError:
            offset = 0

        self._reserve_session(session_id, rollout_path)

        data = self._load_map()
        panes = data.setdefault("panes", {})
        sessions = data.setdefault("sessions", {})

        panes[pane_id] = {
            "session_id": session_id,
            "rollout_path": str(rollout_path),
            "offset": int(offset),
        }
        sessions[session_id] = {
            "pane_id": pane_id,
            "rollout_path": str(rollout_path),
            "offset": int(offset),
        }
        self._write_map(data)

    def consume_session_until_eof(self, session_id: str) -> None:
        data = self._load_map()
        entry = data.get("sessions", {}).get(session_id)
        if entry is None:
            return
        rollout_path = Path(entry.get("rollout_path", ""))
        if not rollout_path.exists():
            return
        try:
            size = rollout_path.stat().st_size
        except OSError:
            return
        entry["offset"] = int(size)
        sessions = data.setdefault("sessions", {})
        sessions[session_id] = entry
        panes = data.setdefault("panes", {})
        for pane_id, pane_entry in panes.items():
            if pane_entry.get("session_id") == session_id:
                pane_entry["offset"] = int(size)
        self._write_map(data)

    def refresh_session_id(self, session_id: str) -> str:
        data = self._load_map()
        sessions = data.get("sessions", {})
        entry = sessions.get(session_id)
        if entry is None or not session_id.startswith("unknown-"):
            return session_id

        rollout_path = Path(entry.get("rollout_path", ""))
        actual_id = self._extract_session_meta(rollout_path)
        if not actual_id or actual_id.startswith("unknown-"):
            return session_id

        entry["session_id"] = actual_id
        sessions.pop(session_id, None)
        sessions[actual_id] = entry

        panes = data.get("panes", {})
        for pane_entry in panes.values():
            if pane_entry.get("session_id") == session_id:
                pane_entry["session_id"] = actual_id

        self._write_map(data)

        if session_id in self._owned_sessions:
            self._owned_sessions.discard(session_id)
            self._owned_sessions.add(actual_id)
        if session_id in self._forced_done:
            self._forced_done.discard(session_id)
            self._forced_done.add(actual_id)

        if self._registry_dir.exists():
            old_record = self._registry_dir / f"{session_id}.json"
            new_record = self._registry_dir / f"{actual_id}.json"
            if old_record.exists():
                try:
                    old_record.rename(new_record)
                except OSError:
                    pass

        return actual_id

    def bind_existing_session(self, *, pane_id: str, session_id: str) -> None:
        data = self._load_map()
        sessions = data.setdefault("sessions", {})
        entry = sessions.get(session_id)
        if entry is None:
            raise RuntimeError(f"Session {session_id!r} not found in session_map")

        rollout_path = Path(entry.get("rollout_path", ""))
        try:
            offset = rollout_path.stat().st_size
        except OSError:
            offset = int(entry.get("offset", 0))

        entry["pane_id"] = pane_id
        entry["offset"] = int(offset)

        panes = data.setdefault("panes", {})
        for existing_pane, pane_entry in list(panes.items()):
            if existing_pane == pane_id or pane_entry.get("session_id") == session_id:
                panes.pop(existing_pane, None)

        panes[pane_id] = {
            "session_id": session_id,
            "rollout_path": entry["rollout_path"],
            "offset": int(offset),
        }

        self._write_map(data)

    def snapshot_rollouts(self) -> Dict[Path, float]:
        if not self.codex_sessions_root.exists():
            return {}
        result: Dict[Path, float] = {}
        for path in self.codex_sessions_root.glob("**/rollout-*.jsonl"):
            try:
                result[path] = path.stat().st_mtime
            except FileNotFoundError:
                continue
        return result

    def register_new_rollout(
        self,
        *,
        pane_id: str,
        baseline: Mapping[Path, float],
        timeout_seconds: float = 30.0,
    ) -> str:
        baseline_map: Dict[Path, float] = dict(baseline)
        deadline = time.time() + timeout_seconds

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            paths = self._wait_for_new_rollouts(
                baseline_map,
                expected=1,
                timeout_seconds=remaining,
            )
            if not paths:
                break
            for rollout_path in paths:
                self._mark_rollout_seen(baseline_map, rollout_path)
                session_id = self._parse_session_meta(rollout_path)
                try:
                    self.register_session(
                        pane_id=pane_id,
                        session_id=session_id,
                        rollout_path=rollout_path,
                    )
                except SessionReservationError:
                    continue
                return session_id

        raise TimeoutError("Failed to detect available Codex session rollout")

    def register_worker_rollouts(
        self,
        *,
        worker_panes: Sequence[str],
        baseline: Mapping[Path, float],
        timeout_seconds: float = 30.0,
    ) -> Dict[str, str]:
        if not worker_panes:
            return {}

        baseline_map: Dict[Path, float] = dict(baseline)
        deadline = time.time() + timeout_seconds
        fork_map: Dict[str, str] = {}

        while len(fork_map) < len(worker_panes):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            paths = self._wait_for_new_rollouts(
                baseline_map,
                expected=1,
                timeout_seconds=remaining,
            )
            if not paths:
                break
            for path in paths:
                self._mark_rollout_seen(baseline_map, path)
                pane_index = len(fork_map)
                if pane_index >= len(worker_panes):
                    break
                pane_id = worker_panes[pane_index]
                session_id = self._parse_session_meta(path)
                try:
                    self.register_session(pane_id=pane_id, session_id=session_id, rollout_path=path)
                except SessionReservationError:
                    continue
                fork_map[pane_id] = session_id
                if len(fork_map) == len(worker_panes):
                    break

        if len(fork_map) < len(worker_panes):
            raise TimeoutError(f"Detected {len(fork_map)} worker rollouts but {len(worker_panes)} required.")

        return fork_map

    def get_last_assistant_message(self, session_id: str) -> Optional[str]:
        data = self._load_map()
        sessions = data.get("sessions", {})
        entry = sessions.get(session_id)
        if entry is None:
            return None

        rollout_path = Path(entry.get("rollout_path", ""))
        if not rollout_path.exists():
            return None

        last_text: Optional[str] = None
        try:
            with rollout_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if obj.get("type") != "response_item":
                        continue

                    payload = obj.get("payload", {})
                    if payload.get("role") != "assistant":
                        continue

                    texts: List[str] = []
                    for block in payload.get("content", []):
                        block_type = block.get("type")
                        if block_type in {"output_text", "text"}:
                            texts.append(block.get("text", ""))
                        elif block_type == "output_markdown":
                            texts.append(block.get("markdown", ""))
                        elif block_type == "output_json":
                            data = block.get("json")
                            if data is not None:
                                texts.append(json.dumps(data))
                    if texts:
                        last_text = "\n".join(part for part in texts if part).strip()
        except OSError:
            return None

        return last_text

    def capture_instruction(self, *, pane_id: str, instruction: str) -> str:
        data = self._load_map()
        pane_entry = data.get("panes", {}).get(pane_id)
        if pane_entry is None:
            raise RuntimeError(
                f"Pane {pane_id!r} is not registered in session_map; ensure Codex session detection succeeded."
            )

        instruction_log = self.logs_dir / "instruction.log"
        with instruction_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"pane": pane_id, "instruction": instruction}) + "\n")

        return pane_entry["session_id"]

    def await_completion(
        self,
        *,
        session_ids: Iterable[str],
        timeout_seconds: Optional[int] = None,
        signal_paths: Optional[Mapping[str, Union[str, Path]]] = None,
    ) -> Dict[str, Any]:
        data = self._load_map()
        sessions = data.get("sessions", {})

        targets: Dict[str, Path] = {}
        offsets: Dict[str, int] = {}
        for session_id in session_ids:
            entry = sessions.get(session_id)
            if entry is None:
                raise RuntimeError(f"Session {session_id!r} not found in session_map")
            targets[session_id] = Path(entry["rollout_path"])
            offsets[session_id] = int(entry.get("offset", 0))

        remaining = set(targets)
        completion: Dict[str, Any] = {}
        signal_targets: Dict[str, Path] = {}
        if signal_paths:
            for session_id, raw_path in signal_paths.items():
                if not session_id:
                    continue
                flag_path = Path(raw_path)
                signal_targets[session_id] = flag_path
                self._active_signal_paths[session_id] = flag_path

        def consume_forced() -> None:
            forced_now = remaining.intersection(self._forced_done)
            for session_id in list(forced_now):
                path = targets[session_id]
                try:
                    offset = path.stat().st_size
                except OSError:
                    offset = 0
                completion[session_id] = {"done": True, "rollout_path": str(path), "forced": True}
                offsets[session_id] = offset
                remaining.remove(session_id)

        deadline = None if timeout_seconds is None else time.time() + timeout_seconds
        while remaining:
            consume_forced()
            if not remaining:
                break
            for session_id in list(remaining):
                if session_id in signal_targets and signal_targets[session_id].exists():
                    completion[session_id] = {"done": True, "rollout_path": str(targets[session_id])}
                    remaining.remove(session_id)
                    flag_path = signal_targets[session_id]
                    try:
                        flag_path.unlink()
                    except OSError:
                        pass
                    continue
                done, new_offset = self._contains_done(
                    session_id=session_id,
                    rollout_path=targets[session_id],
                    offset=offsets.get(session_id, 0),
                )
                if new_offset != offsets.get(session_id, 0):
                    offsets[session_id] = new_offset
                    self._update_session_offset(session_id, new_offset)
                if done:
                    completion[session_id] = {"done": True, "rollout_path": str(targets[session_id])}
                    remaining.remove(session_id)
            if not remaining:
                break
            if deadline is not None and time.time() >= deadline:
                break
            time.sleep(self.poll_interval)

        for session_id in remaining:
            completion[session_id] = {
                "done": False,
                "rollout_path": str(targets[session_id]),
            }

        for session_id in signal_targets:
            self._active_signal_paths.pop(session_id, None)

        return completion

    def force_completion(self, session_ids: Iterable[str]) -> None:
        for session_id in session_ids:
            if session_id:
                self._forced_done.add(session_id)
                self._release_session(session_id)

    def wait_for_rollout_activity(
        self,
        session_id: str,
        *,
        min_bytes: int = 1,
        timeout_seconds: float = 5.0,
    ) -> None:
        data = self._load_map()
        sessions = data.get("sessions", {})
        entry = sessions.get(session_id)
        if entry is None:
            return
        rollout_path = Path(entry.get("rollout_path", ""))
        baseline = int(entry.get("offset", 0))
        deadline = time.time() + timeout_seconds
        last_size = baseline

        while time.time() < deadline:
            try:
                size = rollout_path.stat().st_size
            except OSError:
                break
            if size - baseline >= min_bytes:
                last_size = size
                break
            time.sleep(self.poll_interval / 2)
        entry["offset"] = int(last_size)
        self._update_session_offset(session_id, int(last_size))

    # 内部ユーティリティ  -------------------------------------------------
    def _reserve_session(self, session_id: str, rollout_path: Path) -> None:
        try:
            self._registry_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        metadata = {
            "session_id": session_id,
            "namespace": self._session_namespace,
            "pid": os.getpid(),
            "timestamp": time.time(),
            "rollout_path": str(rollout_path),
        }
        record_path = self._registry_dir / f"{session_id}.json"
        while True:
            try:
                with record_path.open("x", encoding="utf-8") as fh:
                    json.dump(metadata, fh, ensure_ascii=False)
                self._owned_sessions.add(session_id)
                return
            except FileExistsError:
                try:
                    existing = json.loads(record_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    existing = {}
                owner_ns = existing.get("namespace")
                owner_pid = existing.get("pid")
                if owner_ns == self._session_namespace:
                    try:
                        record_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
                        self._owned_sessions.add(session_id)
                    except OSError:
                        pass
                    return
                if owner_pid and not self._pid_exists(owner_pid):
                    try:
                        record_path.unlink()
                    except OSError:
                        break
                    continue
                raise SessionReservationError(session_id, owner_ns)
            except OSError:
                return

    def _release_session(self, session_id: str) -> None:
        record_path = self._registry_dir / f"{session_id}.json"
        try:
            existing = json.loads(record_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self._owned_sessions.discard(session_id)
            return
        if existing.get("namespace") == self._session_namespace:
            try:
                record_path.unlink()
            except OSError:
                pass
            self._owned_sessions.discard(session_id)

    @staticmethod
    def _pid_exists(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _mark_rollout_seen(self, baseline: MutableMapping[Path, float], path: Path) -> None:
        try:
            baseline[path] = path.stat().st_mtime
        except OSError:
            baseline[path] = time.time()

    def _wait_for_new_rollouts(
        self,
        baseline: Mapping[Path, float],
        *,
        expected: int,
        timeout_seconds: float,
    ) -> List[Path]:
        deadline = time.time() + timeout_seconds
        baseline_paths = set(baseline.keys())
        while True:
            current = self.snapshot_rollouts()
            new_paths = [path for path in current.keys() if path not in baseline_paths]
            if len(new_paths) >= expected:
                new_paths.sort(key=lambda p: current.get(p, 0.0))
                return new_paths
            if time.time() >= deadline:
                new_paths.sort(key=lambda p: current.get(p, 0.0))
                return new_paths
            time.sleep(self.poll_interval)

    def _parse_session_meta(self, rollout_path: Path) -> str:
        session_id = self._extract_session_meta(rollout_path)
        if session_id:
            return session_id
        suffix = int(time.time() * 1000)
        return f"unknown-{suffix}"

    def _extract_session_meta(self, rollout_path: Path) -> Optional[str]:
        try:
            with rollout_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") == "session_meta" and "payload" in obj:
                        ident = obj["payload"].get("id")
                        if ident:
                            return str(ident)
        except FileNotFoundError:
            pass
        return None

    def _load_map(self) -> Dict[str, Any]:
        text = self.session_map_path.read_text(encoding="utf-8")
        if not text.strip():
            return {}
        return yaml.safe_load(text) or {}

    def _write_map(self, data: Mapping[str, Any]) -> None:
        self.session_map_path.write_text(yaml.safe_dump(dict(data), sort_keys=True), encoding="utf-8")

    def _contains_done(
        self,
        *,
        session_id: str,
        rollout_path: Path,
        offset: int,
    ) -> tuple[bool, int]:
        if not rollout_path.exists():
            return False, offset
        try:
            with rollout_path.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read()
                new_offset = fh.tell()
        except OSError:
            return False, offset

        if not chunk:
            return False, new_offset

        done_detected = False
        for line in chunk.decode("utf-8", errors="ignore").splitlines():
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("type") != "response_item":
                continue

            payload = obj.get("payload", {})
            if payload.get("role") != "assistant":
                continue

            for block in payload.get("content", []):
                block_type = block.get("type")
                if block_type in {"output_text", "text"}:
                    text = block.get("text", "")
                    lines = [segment.strip() for segment in text.splitlines() if segment.strip()]
                    if any(segment == "/done" for segment in lines):
                        done_detected = True
                        break
            if done_detected:
                break

        return done_detected, new_offset

    def _update_session_offset(self, session_id: str, new_offset: int) -> None:
        data = self._load_map()
        sessions = data.get("sessions", {})
        panes = data.get("panes", {})
        session_entry = sessions.get(session_id)
        if session_entry is not None:
            session_entry["offset"] = int(new_offset)
            pane_id = session_entry.get("pane_id")
            if pane_id and pane_id in panes:
                panes[pane_id]["offset"] = int(new_offset)
            self._write_map(data)
