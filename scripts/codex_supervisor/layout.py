"""Keep machine artifacts below one directory; accept legacy archives for offline tools."""

from pathlib import Path


def artifacts(run: Path) -> Path:
    run = run.resolve()
    return run / "artifacts" if (run / "artifacts").is_dir() else run


def create_run(run: Path) -> Path:
    run.mkdir(parents=True, exist_ok=False)
    directory = run / "artifacts"
    directory.mkdir()
    for name in ("rollout.mp4", "memory.md", "review.md"):
        (run / name).symlink_to(Path("artifacts") / name)
    return directory
