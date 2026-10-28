"""Duration statistics must count physical timestep gaps once, including the final control."""

import json

import pytest

from scripts.build_scheduling_priors import duration_groups
from scripts.build_scheduling_priors import import_experience
from scripts.build_scheduling_priors import percentile
from scripts.build_scheduling_priors import read_samples


def test_dedup_and_segments_exclude_demo_offset(tmp_path):
    source = tmp_path / "rows.jsonl"
    rows = []
    for step, label in [(50, "pick"), (60, "pick"), (60, "pick"), (75, "place"), (84, "place")]:
        rows.append({"images": [f"Task_ep0_step{step}.png"], "messages": [{"role": "assistant", "content": label}]})
    source.write_text("\n".join(json.dumps(row) for row in rows))
    episodes, tasks, duplicates = read_samples(source)
    groups = duration_groups(episodes)
    assert tasks == ["Task"]
    assert duplicates == 1
    assert groups["Task", "pick"][0]["steps"] == 25
    assert groups["Task", "place"][0]["steps"] == 10
    assert percentile([100, 110, 120], 0.25) == 105


def test_conflicting_training_rows_are_rejected(tmp_path):
    source = tmp_path / "rows.jsonl"
    source.write_text(
        "\n".join(
            json.dumps({"images": ["Task_ep0_step0.png"], "messages": [{"role": "assistant", "content": label}]})
            for label in ["pick", "place"]
        )
    )
    with pytest.raises(ValueError, match="conflicting"):
        read_samples(source)


def test_migration_retains_common_facts_and_isolates_legacy_budget(tmp_path):
    source = tmp_path / "Task.json"
    entry = {
        "id": "one",
        "claim": "Measured motion stalled",
        "conditions": "one observation",
        "supervision": {"recommended_chunks": 1},
    }
    source.write_text(json.dumps({"entries": [entry]}))
    before = source.read_bytes()
    libraries, mapping = import_experience([source, source], {})
    common = next(e for e in libraries["Task"] if e["applicability"] == "common")
    assert common["claim"] == entry["claim"]
    assert "supervision" not in common
    assert len(libraries["Task"]) == 2
    assert source.read_bytes() == before
    assert mapping[-1]["duplicate_of"] == "one"
