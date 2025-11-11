from pathlib import Path

import git
import pytest

from parallel_developer.services import WorktreeManager


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = git.Repo.init(tmp_path, initial_branch="main")
    readme = tmp_path / "README.md"
    readme.write_text("# demo\n", encoding="utf-8")
    repo.index.add([str(readme)])
    repo.index.commit("Initial commit")
    return tmp_path


def test_worktree_manager_creates_isolated_worktrees(git_repo: Path):
    manager = WorktreeManager(root=git_repo, worker_count=2, session_namespace="session-a")

    mapping = manager.prepare()

    assert len(mapping) == 2
    worker_dirs = list(mapping.values())
    for path in worker_dirs:
        assert path.is_dir()
        assert (path / ".git").exists() or (path / ".git").is_file()
        repo = git.Repo(path)
        assert repo.head.commit.hexsha == git.Repo(git_repo).head.commit.hexsha

    base = git_repo / ".parallel-dev" / "sessions" / "session-a" / "worktrees"
    expected = [base / "worker-1", base / "worker-2"]
    assert sorted(worker_dirs) == sorted(expected)
    assert manager.boss_path == base / "boss"
    assert manager.boss_path.is_dir()



def test_worktree_manager_requires_initial_commit(tmp_path: Path):
    repo = git.Repo.init(tmp_path, initial_branch="main")
    with pytest.raises(RuntimeError):
        WorktreeManager(root=tmp_path, worker_count=1, session_namespace="session-a")


def test_worktree_manager_separates_sessions(git_repo: Path):
    manager_a = WorktreeManager(root=git_repo, worker_count=1, session_namespace="session-a")
    mapping_a = manager_a.prepare()
    worker_a = mapping_a["worker-1"]
    (worker_a / "keep.txt").write_text("session a", encoding="utf-8")

    manager_b = WorktreeManager(root=git_repo, worker_count=1, session_namespace="session-b")
    mapping_b = manager_b.prepare()
    worker_b = mapping_b["worker-1"]

    assert worker_a != worker_b
    assert (worker_a / "keep.txt").read_text(encoding="utf-8") == "session a"
