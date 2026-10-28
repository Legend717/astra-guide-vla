from pathlib import Path

from scripts.evaluate_release import ROOT
from scripts.evaluate_release import build_plan


def test_paper_schedule_and_shared_experience(tmp_path):
    order, plan = build_plan(ROOT, tmp_path, Path("/sim/python"), "127.0.0.1", 8000)
    assert len(order) == 16
    assert len(plan) == len({(r["task"], r["episode"]) for r in plan}) == 800
    for item in plan:
        command = item["command"]
        policy = "always" if item["episode"] < 5 else "failure-only"
        assert command[command.index("--experience-update-policy") + 1] == policy
        assert command[command.index("--experience") + 1] == str(tmp_path / "experience/general.json")


def test_task_filter_keeps_all_fifty_episodes(tmp_path):
    order, plan = build_plan(ROOT, tmp_path, Path("/sim/python"), "127.0.0.1", 8000, ["InsertPeg"])
    assert order == ["InsertPeg"]
    assert [r["episode"] for r in plan] == list(range(50))


def test_final_experience_plan_is_frozen_and_passes_existing_environment(tmp_path):
    _, plan = build_plan(
        ROOT,
        tmp_path,
        Path("/existing/sim/python"),
        "127.0.0.1",
        8000,
        experience_updates="frozen",
        env_file=Path("/existing/.env"),
        cpu_render_launcher=Path("/existing/render.sh"),
    )
    assert len(plan) == 800
    for item in plan:
        command = item["command"]
        assert "--no-experience-update" in command
        assert "--experience-update" not in command
        assert command[command.index("--env-file") + 1] == "/existing/.env"
        assert command[command.index("--cpu-render-launcher") + 1] == "/existing/render.sh"


def test_initial_library_rejects_learned_entries(tmp_path):
    import json

    import pytest

    from scripts.evaluate_release import validate_experience

    (tmp_path / "general.json").write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [{"id": "learned", "applicability": "common", "claim": "c", "conditions": "c"}],
            }
        )
    )
    with pytest.raises(ValueError, match="bundled baseline"):
        validate_experience(tmp_path, [], initial_preset=True)


def test_library_rejects_jev_and_duplicate_ids(tmp_path):
    import json

    import pytest

    from scripts.evaluate_release import validate_experience

    entry = {"id": "entry", "applicability": "jev", "claim": "c", "conditions": "c"}
    path = tmp_path / "general.json"
    path.write_text(json.dumps({"version": 1, "entries": [entry]}))
    with pytest.raises(ValueError, match="Non-step-budget"):
        validate_experience(tmp_path, [])
    entry["applicability"] = "common"
    path.write_text(json.dumps({"version": 1, "entries": [entry, entry]}))
    with pytest.raises(ValueError, match="duplicate"):
        validate_experience(tmp_path, [])


def test_packaged_experience_snapshots():
    import json
    import re

    from scripts.evaluate_release import TASKS
    from scripts.evaluate_release import validate_experience

    initial = ROOT / "assets/initial-experience"
    final = ROOT / "assets/final-experience"
    assert all(validate_experience(initial, TASKS, initial_preset=True).values())
    assert all(validate_experience(final, TASKS).values())
    baseline = json.loads((initial / "general.json").read_text())["entries"]
    final_global = json.loads((final / "general.json").read_text())["entries"]
    assert baseline
    assert all(entry in final_global for entry in baseline)
    for path in [final / "general.json", *sorted((final / "tasks").glob("*.json"))]:
        text = path.read_text()
        assert not re.search(r"[\u4e00-\u9fff]|/robochallenge/|jev|user-curation", text, re.IGNORECASE)
        data = json.loads(text)
        assert not any(e.get("supersedes") or e.get("derived_from") for e in data["entries"])


def test_initial_task_libraries_reject_extra_guidance(tmp_path):
    import json
    import shutil

    import pytest

    from scripts.evaluate_release import validate_experience

    shutil.copyfile(ROOT / "assets/initial-experience/general.json", tmp_path / "general.json")
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks/InsertPeg.json").write_text(
        json.dumps(
            {
                "version": 1,
                "entries": [{"id": "task-tip", "applicability": "common", "claim": "c", "conditions": "c"}],
            }
        )
    )
    with pytest.raises(ValueError, match="training priors only"):
        validate_experience(tmp_path, ["InsertPeg"], initial_preset=True)


def test_selected_episodes_preserve_update_policy(tmp_path):
    import pytest

    _, plan = build_plan(
        ROOT,
        tmp_path,
        Path("/sim/python"),
        "127.0.0.1",
        8000,
        ["InsertPeg", "StopCube"],
        episodes=[0, 4, 5],
    )
    assert [(item["task"], item["episode"]) for item in plan] == [
        (task, episode) for task in ["InsertPeg", "StopCube"] for episode in [0, 4, 5]
    ]
    for item in plan:
        command = item["command"]
        assert command[command.index("--experience-update-policy") + 1] == (
            "always" if item["episode"] < 5 else "failure-only"
        )
    for episodes in ([], [0, 0], [-1], [50]):
        with pytest.raises(ValueError, match="Episodes"):
            build_plan(ROOT, tmp_path, Path("/sim/python"), "127.0.0.1", 8000, episodes=episodes)
