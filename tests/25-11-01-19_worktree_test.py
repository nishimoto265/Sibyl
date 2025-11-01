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
    manager = WorktreeManager(root=git_repo, worker_count=2)

    mapping = manager.prepare()

    assert len(mapping) == 2
    worker_dirs = list(mapping.values())
    for path in worker_dirs:
        assert path.is_dir()
        assert (path / ".git").exists() or (path / ".git").is_file()
        repo = git.Repo(path)
        assert repo.head.commit.hexsha == git.Repo(git_repo).head.commit.hexsha

    expected = [
        git_repo / ".parallel-dev" / "worktrees" / "worker-1",
        git_repo / ".parallel-dev" / "worktrees" / "worker-2",
    ]
    assert sorted(worker_dirs) == sorted(expected)
    assert manager.boss_path.is_dir()


def test_merge_into_main_fast_forward(git_repo: Path):
    manager = WorktreeManager(root=git_repo, worker_count=1)
    mapping = manager.prepare()

    worker_path = mapping["worker-1"]
    worker_repo = git.Repo(worker_path)
    new_file = worker_path / "feature.txt"
    new_file.write_text("feature branch change\n", encoding="utf-8")
    worker_repo.index.add([str(new_file)])
    worker_repo.index.commit("Add feature file")

    manager.merge_into_main(manager.worker_branch("worker-1"))

    main_repo = git.Repo(git_repo)
    assert (git_repo / "feature.txt").exists()
    assert main_repo.is_dirty() is False


def test_worktree_manager_requires_initial_commit(tmp_path: Path):
    repo = git.Repo.init(tmp_path, initial_branch="main")
    with pytest.raises(RuntimeError):
        WorktreeManager(root=tmp_path, worker_count=1)
