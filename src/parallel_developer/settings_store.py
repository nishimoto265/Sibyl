"""Persistence helper for CLI settings stored in the user configuration directory."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import yaml
from platformdirs import PlatformDirs

CONFIG_FILENAME = "config.yaml"
ENV_CONFIG_PATH = "PARALLEL_DEV_CONFIG_PATH"
ENV_WORKTREE_ROOT = "PARALLEL_DEV_WORKTREE_ROOT"
_UNSET = object()


def default_config_dir() -> Path:
    system = platform.system().lower()
    if "windows" in system:
        dirs = PlatformDirs(appname="ParallelDeveloper", appauthor="ParallelDeveloper", roaming=True)
        return Path(dirs.user_config_path)
    return Path.home() / ".parallel-dev"


def resolve_settings_path(explicit_path: Optional[Path] = None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path)
    env_path = os.getenv(ENV_CONFIG_PATH)
    if env_path:
        return Path(env_path).expanduser()
    return default_config_dir() / CONFIG_FILENAME


def resolve_worktree_root(config_value: Optional[str], fallback: Path) -> Path:
    env_value = os.getenv(ENV_WORKTREE_ROOT)
    if env_value:
        return Path(env_value).expanduser()
    if config_value:
        return Path(config_value).expanduser()
    return Path(fallback)


@dataclass
class SettingsData:
    attach: str = "auto"
    boss: str = "score"
    flow: str = "manual"
    parallel: str = "3"
    mode: str = "parallel"
    commit: str = "manual"
    worktree_root: Optional[str] = None


class SettingsStore:
    """Load and persist CLI configuration flags."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: SettingsData = self._load()

    @property
    def attach(self) -> str:
        return self._data.attach

    @attach.setter
    def attach(self, value: str) -> None:
        self._data.attach = value
        self._save()

    @property
    def boss(self) -> str:
        return self._data.boss

    @boss.setter
    def boss(self, value: str) -> None:
        self._data.boss = value
        self._save()

    @property
    def flow(self) -> str:
        return self._data.flow

    @flow.setter
    def flow(self, value: str) -> None:
        self._data.flow = value
        self._save()

    @property
    def parallel(self) -> str:
        return self._data.parallel

    @parallel.setter
    def parallel(self, value: str) -> None:
        self._data.parallel = value
        self._save()

    @property
    def mode(self) -> str:
        return self._data.mode

    @mode.setter
    def mode(self, value: str) -> None:
        self._data.mode = value
        self._save()

    @property
    def commit(self) -> str:
        return self._data.commit

    @commit.setter
    def commit(self, value: str) -> None:
        self._data.commit = value
        self._save()

    @property
    def worktree_root(self) -> Optional[str]:
        return self._data.worktree_root

    @worktree_root.setter
    def worktree_root(self, value: Optional[object]) -> None:
        self._data.worktree_root = str(value) if value else None
        self._save()

    def snapshot(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "commands": {
                "attach": self._data.attach,
                "boss": self._data.boss,
                "flow": self._data.flow,
                "parallel": self._data.parallel,
                "mode": self._data.mode,
                "commit": self._data.commit,
            }
        }
        if self._data.worktree_root:
            payload["paths"] = {"worktree_root": self._data.worktree_root}
        return payload

    def update(
        self,
        *,
        attach: Optional[str] = None,
        boss: Optional[str] = None,
        flow: Optional[str] = None,
        parallel: Optional[str] = None,
        mode: Optional[str] = None,
        commit: Optional[str] = None,
        worktree_root: object = _UNSET,
    ) -> None:
        if attach is not None:
            self._data.attach = attach
        if boss is not None:
            self._data.boss = boss
        if flow is not None:
            self._data.flow = flow
        if parallel is not None:
            self._data.parallel = parallel
        if mode is not None:
            self._data.mode = mode
        if commit is not None:
            self._data.commit = commit
        if worktree_root is not _UNSET:
            self._data.worktree_root = str(worktree_root) if worktree_root else None
        self._save()

    def _load(self) -> SettingsData:
        payload: Dict[str, object]
        if self._path.exists():
            try:
                payload = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError:
                payload = {}
        else:
            payload = {}

        commands = payload.get("commands") if isinstance(payload, dict) else None

        paths_data = payload.get("paths") if isinstance(payload, dict) else None
        worktree_root_value: Optional[str] = None
        if isinstance(paths_data, dict):
            raw_root = paths_data.get("worktree_root")
            if raw_root:
                worktree_root_value = str(raw_root)
        if isinstance(commands, dict):
            return SettingsData(
                attach=str(commands.get("attach", "auto")),
                boss=str(commands.get("boss", "score")),
                flow=str(commands.get("flow", "manual")),
                parallel=str(commands.get("parallel", "3")),
                mode=str(commands.get("mode", "parallel")),
                commit=str(commands.get("commit", "manual")),
                worktree_root=worktree_root_value,
            )

        # Legacy YAML keys fallback
        return SettingsData(
            attach=str(payload.get("attach_mode", "auto")),
            boss=str(payload.get("boss_mode", "score")),
            flow=str(payload.get("flow_mode", "manual")),
            parallel=str(payload.get("worker_count", "3")),
            mode=str(payload.get("session_mode", "parallel")),
            commit="auto" if bool(payload.get("auto_commit", False)) else "manual",
            worktree_root=None,
        )

    def _save(self) -> None:
        try:
            self._path.write_text(
                yaml.safe_dump(self.snapshot(), sort_keys=True, allow_unicode=True),
                encoding="utf-8",
            )
        except OSError:
            pass
