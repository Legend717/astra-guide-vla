"""Protocol and information-boundary tests; real simulator checks are separately archived."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from scripts import codex_rgb_eval as rgb
from scripts import codex_symbolic_eval as cli
from scripts.codex_supervisor import experience
from scripts.codex_supervisor.core import Supervisor
from scripts.codex_supervisor.instructions import TEMPLATES
from scripts.codex_supervisor.instructions import convert

SPEC = {
    "text": "pick up the red cube at <point> for the first time",
    "point_yx_1000": [[500, 250]],
    "simple_subgoal": "pick up the red cube for the first time",
}


def test_history_keeps_every_frame_and_both_cameras(tmp_path):
    from PIL import Image

    from scripts.codex_supervisor.history import render_history

    for index in range(10):
        paths = {}
        for camera, color in (("front", (index, 0, 0)), ("wrist", (0, index, 0))):
            path = tmp_path / f"{index}-{camera}.png"
            Image.new("RGB", (256, 256), color).save(path)
            paths[camera] = path.name
        rgb.append_json(tmp_path / "frames.jsonl", {"frame_id": index, "rgb": paths})
    result = render_history(tmp_path, 1, 9)
    assert [frame["frame_id"] for frame in result["frames"]] == list(range(1, 10))
    assert len(result["contact_sheets"]) == 2
    with Image.open(tmp_path / result["contact_sheets"][-1]) as sheet:
        assert sheet.size == (1024, 280)
        assert sheet.getpixel((100, 100))[0] > sheet.getpixel((100, 100))[1]
        assert sheet.getpixel((350, 100))[1] > sheet.getpixel((350, 100))[0]
    with pytest.raises(ValueError, match="missing frames"):
        render_history(tmp_path, 1, 10)
    with pytest.raises(ValueError, match="invalid"):
        render_history(tmp_path, 9, 1)


def test_request_history_excludes_previous_frames(tmp_path, capsys):
    rgb.write_json(tmp_path / "latest.json", {"frame_id": 3})
    rgb.append_json(tmp_path / "events.jsonl", {"event": "release", "request_id": "one", "frame": 3})
    rgb.append_json(tmp_path / "events.jsonl", {"event": "execution_end", "request_id": "one", "after_frame": 3})
    args = SimpleNamespace(run_dir=tmp_path, request_id="one", from_frame=None, to_frame=None)
    cli.history(args)
    assert json.loads(capsys.readouterr().out)["frame_count"] == 0
    args.request_id = "pending"
    with pytest.raises(ValueError, match="completed"):
        cli.history(args)


def test_preflight_preserves_virtualenv_python_symlink(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "checkpoint/79999"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets/robomme").mkdir(parents=True)
    (checkpoint / "assets/robomme/norm_stats.json").write_text("{}")
    (checkpoint.parent / "history_config.txt").write_text("symbolic-grounded-subgoal.yaml\n")
    base_python = tmp_path / "base-python"
    base_python.touch()
    venv_python = tmp_path / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)
    commands = []

    def probe(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(stdout='{"ok": true}\n', returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", probe)
    cli.preflight(SimpleNamespace(checkpoint=checkpoint, policy_python=venv_python, mode="grounded"))
    assert commands[1][0] == str(venv_python)
    assert json.loads(capsys.readouterr().out)["policy"]["ok"]


@pytest.mark.parametrize(
    ("point", "expected"),
    [([0, 0], [0, 0]), ([1000, 1000], [479, 639]), ([500, 250], [240, 160]), ([999, 999], [479, 639])],
)
def test_coordinates_use_original_dimensions(point, expected):
    converted = convert("PickXtimes", {**SPEC, "point_yx_1000": [point]}, (480, 640))
    assert converted["pixel_yx"] == [expected]
    assert f"<{expected[0]}, {expected[1]}>" in converted["grounded_subgoal"]


@pytest.mark.parametrize(
    "points", [[], [[-1, 0]], [[0, 1001]], [[True, 0]], [[float("nan"), 0]], [[0]], [[0, 0], [0, 0]], "500,500"]
)
def test_bad_coordinates_rejected(points):
    with pytest.raises(ValueError, match="point|template|prefix|request ID"):
        convert("PickXtimes", {**SPEC, "point_yx_1000": points}, (256, 256))


def test_multiple_points_and_no_point():
    result = convert(
        "MoveCube",
        {
            "text": "hook the cube at <point> to the target at <point> with the peg",
            "point_yx_1000": [[0, 1000], [1000, 0]],
        },
        (256, 512),
    )
    assert result["grounded_subgoal"] == "hook the cube at <0, 511> to the target at <255, 0> with the peg"
    assert convert("StopCube", {"text": "remain static"}, (256, 256))["pixel_yx"] == []
    assert convert("PatternLock", {"text": "move forward-left"}, (256, 256))["grounded_subgoal"] == "move forward-left"
    with pytest.raises(ValueError, match="point|template|prefix|request ID"):
        convert("StopCube", {"text": "remain static", "point_yx_1000": [[0, 0]]}, (256, 256))
    with pytest.raises(ValueError, match="point|template|prefix|request ID"):
        convert("PickXtimes", {"text": "find the hidden answer"}, (256, 256))


def test_all_task_templates_can_be_filled():
    import re

    assert set(TEMPLATES) == set(rgb.TASKS)
    for task, forms in TEMPLATES.items():
        for template in forms["grounded"]:
            text = re.sub(r"\[([^]]+)\]", lambda m: "red" if m[1] == "color" else m[1].split("/")[0], template)
            text = text.replace("<y, x>", "<point>")
            convert(task, {"text": text, "point_yx_1000": [[500, 500]] * text.count("<point>")}, (256, 256))


@pytest.fixture
def system(tmp_path):
    (tmp_path / "internal").mkdir()

    class Recorder:
        frame_count = 1
        latest: ClassVar[dict] = {"frame_id": 0, "robot": {"eef_state": [0.0] * 6}, "task_goal": "Official task"}

        def capture(self, obs, info, batch, feedback):
            if obs:
                self.frame_count += 1
                self.latest = {"frame_id": self.frame_count - 1, "robot": {"eef_state": [0.0] * 6}}

    class Runtime:
        calls: ClassVar[list] = []
        status = "ongoing"

        def image_hw(self):
            return 256, 256

        def observation(self, subgoal):
            return {"prompt": "Official task", **subgoal}

        def step(self, action, kind):
            self.calls.append((kind, action))
            if self.status == "ik":
                return (
                    {},
                    987654321,
                    True,
                    False,
                    {
                        "status": "error",
                        "error_message": "ee step (rpy): IK failed (status=IK Failed! Distance 0.01 is greater than threshold 0.001., num_solutions=0)",
                    },
                )
            return {"frame": 1}, 987654321, False, False, {"status": self.status, "secret": "HIDDEN"}

    class Policy:
        count = 0

        def validate(self, obs):
            assert obs["prompt"] == "Official task"

        def infer(self, obs):
            self.count += 1
            return {"actions": np.arange(32).reshape(4, 8) / 32}

    class FK:
        def describe(self, actions, current):
            return {"summary": {"steps": len(actions)}, "trajectory": [{"target_xyz_rpy": [0.0] * 6}] * len(actions)}

    return Supervisor(
        tmp_path, Recorder(), Runtime(), Policy(), FK(), task="PickXtimes", mode="simple-smoke", max_steps=20, prefix=3
    )


def request(system, operation, **kwargs):
    return {
        "operation": operation,
        "expected_frame": system.frame,
        "subgoal_version": system.version,
        "candidate_id": (system.candidate or {}).get("candidate_id"),
        **kwargs,
    }


def initialize(system):
    return system.handle(request(system, "prompt", subgoal=SPEC), "initial")


def test_three_choices_and_pause(system):
    initialize(system)
    assert rgb.read_json(system.directory / "decision.json")["candidate"]["eef"]["trajectory"]
    assert system.recorder.frame_count == 1  # FK is visible without executing or creating future observations.
    first_id = system.candidate["candidate_id"]
    assert system.policy.count == 1
    assert not system.runtime.calls
    system.handle(request(system, "prompt", subgoal={**SPEC, "point_yx_1000": [[510, 250]]}), "refresh")
    assert system.policy.count == 2
    assert system.version == 2
    assert system.frame == 0
    assert system.candidate["candidate_id"] != first_id
    assert not system.runtime.calls
    original = system.actions.copy()
    response = system.handle(request(system, "release", prefix=2), "release")
    assert response["executed_steps"] == 2
    assert system.policy.count == 3
    np.testing.assert_equal([a for _, a in system.runtime.calls], original[:2])
    assert system.recorder.video_context == {
        "source": "VLA",
        "request_id": "release",
        "subgoal": system.subgoal["grounded_subgoal"],
    }
    assert all(kind == "joint" for kind, _ in system.runtime.calls)
    assert system.frame == 2
    stale = request(system, "release")
    system.handle(
        request(
            system, "intervene", sequence=[{"target": [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -1], "steps": 2, "interpolate": 2}]
        ),
        "intervene",
    )
    assert system.policy.count == 4
    assert system.runtime.calls[-2:] == [
        ("eef", [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, -1.0]),
        ("eef", [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, -1.0]),
    ]
    with pytest.raises(ValueError, match="stale"):
        system.handle(stale, "old")
    events = [json.loads(line) for line in (system.directory / "events.jsonl").read_text().splitlines()]
    execution = [e for e in events if e["event"] == "execution_end"]
    assert execution[0]["discarded_tail"] == 2
    assert execution[1]["discarded_tail"] == 4
    assert "HIDDEN" not in json.dumps(events)
    assert "987654321" not in json.dumps(events)
    assert "actions" not in json.dumps(system.candidate)


@pytest.mark.parametrize(
    ("field", "value"),
    [("expected_frame", -1), ("expected_frame", False), ("subgoal_version", 0), ("candidate_id", "old")],
)
def test_stale_candidate(system, field, value):
    initialize(system)
    req = request(system, "release")
    req[field] = value
    with pytest.raises(ValueError, match="stale"):
        system.handle(req, "reject")
    assert not system.runtime.calls
    assert system.policy.count == 1


@pytest.mark.parametrize(
    ("status", "expected"),
    [("success", "success"), ("fail", "failure"), ("timeout", "timeout"), ("error", "error"), ("ik", "incomplete")],
)
def test_stop_partial_execution(system, status, expected):
    initialize(system)
    system.runtime.status = status
    result = system.handle(request(system, "intervene", sequence=[{"target": [0.0] * 7, "steps": 3}] * 2), "partial")
    assert result["executed_steps"] == 1
    assert system.state["outcome"] == expected
    assert system.closed == (status != "ik")
    assert system.policy.count == (2 if status == "ik" else 1)
    if status == "ik":
        assert system.frame == 0
        assert system.state["feedback"]["control_error"]["retryable"]


def test_budget_stop_and_no_tail_replay(system):
    initialize(system)
    system.max_steps = 1
    result = system.handle(request(system, "release", prefix=3), "budget")
    assert result["executed_steps"] == 1
    assert system.closed
    assert system.candidate is None
    assert system.state["stop_reason"] == "control_budget"
    assert system.state["outcome"] == "incomplete"


@pytest.mark.parametrize("prefix", [0, -1, 4, True, 1.5])
def test_invalid_prefix_leaves_candidate_usable(system, prefix):
    initialize(system)
    identifier = system.candidate["candidate_id"]
    with pytest.raises(ValueError, match="point|template|prefix|request ID"):
        system.handle(request(system, "release", prefix=prefix), "bad")
    assert system.candidate["candidate_id"] == identifier
    assert not system.runtime.calls


def test_request_immutability_and_offline_queries(tmp_path, capsys):
    (tmp_path / "requests").mkdir()
    (tmp_path / "responses").mkdir()
    body = {"operation": "release", "expected_frame": 8}
    response = cli.immutable_request(tmp_path, "abc", body)
    assert cli.immutable_request(tmp_path, "abc", body) == response
    assert [json.loads(line) for line in (tmp_path / "inputs.jsonl").read_text().splitlines()] == [
        {"request_id": "abc", "request": body}
    ]
    with pytest.raises(ValueError, match="different body"):
        cli.immutable_request(tmp_path, "abc", {**body, "expected_frame": 9})
    with pytest.raises(ValueError, match="point|template|prefix|request ID"):
        cli.immutable_request(tmp_path, "../escape", body)
    rgb.write_json(response, {"ok": True, "executed_steps": 2})
    args = SimpleNamespace(run_dir=tmp_path, request_id="abc")
    cli.query(args)
    cli.query(args)
    assert capsys.readouterr().out.count('"executed_steps": 2') == 2


@pytest.mark.parametrize(("read", "write"), [(False, False), (False, True), (True, False), (True, True)])
def test_experience_gates_and_no_lost_updates(tmp_path, read, write):
    source = tmp_path / "global.json"
    rgb.write_json(source, {"version": 1, "entries": [{"claim": "OLD PRIVATE TEXT"}]})
    run = tmp_path / "run"
    run.mkdir()
    snap = experience.snapshot(run, source, enabled=read)
    initial = (run / "experience-input.json").read_bytes() if read else None
    assert (run / "experience-input.json").exists() == read
    rgb.write_json(run / "manifest.json", {"experience_input": snap, "experience_update": write, "mode": "grounded"})
    rgb.write_json(run / "result.json", {"lifecycle": "closed", "recorded_frames": 3})
    entry = {
        "claim": "New",
        "conditions": "Observed condition",
        "confidence": "low",
        "frames": [1],
        "kind": "hypothesis",
    }
    if not write:
        with pytest.raises(ValueError, match="disabled"):
            experience.commit(run, {"entries": [entry]})
        assert rgb.read_json(source)["version"] == 1
    else:
        receipt = experience.commit(run, {"entries": [entry]})
        assert "OLD PRIVATE TEXT" not in json.dumps(receipt)
        experience.commit(run, {"entries": [entry]})
        assert rgb.read_json(source)["version"] == 2
        assert len(rgb.read_json(source)["entries"]) == 2
        experience.commit(run, {"entries": [{**entry, "claim": "Another"}]})
        assert len(rgb.read_json(source)["entries"]) == 3
    if read:
        assert (run / "experience-input.json").read_bytes() == initial


def test_smoke_experience_cannot_be_task_success(tmp_path):
    rgb.write_json(tmp_path / "manifest.json", {"experience_update": True, "mode": "simple-smoke"})
    rgb.write_json(tmp_path / "result.json", {"lifecycle": "closed", "recorded_frames": 1})
    with pytest.raises(ValueError, match="engineering"):
        experience.commit(
            tmp_path,
            {
                "entries": [
                    {
                        "claim": "Solved",
                        "conditions": "smoke",
                        "confidence": "high",
                        "frames": [0],
                        "kind": "observation",
                    }
                ]
            },
        )


def test_token_budget_checks_exact_tokenizer_without_dependencies():
    # Execute the actual tokenizer class with a counting SentencePiece double. The
    # assertion covers both complete prompt formats, including state and BOS.
    import ast
    import logging

    path = Path(rgb.ROOT / "src/mme_vla_suite/training/config.py")
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "PaligemmaTokenizer"
    )
    scope = {"np": np, "logging": logging}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
    tokenizer = scope["PaligemmaTokenizer"].__new__(scope["PaligemmaTokenizer"])
    tokenizer._max_len = 10  # noqa: SLF001
    tokenizer.reject_truncation = True
    tokenizer._tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text))))  # noqa: SLF001
    for state in (None, np.zeros(8)):
        with pytest.raises(ValueError, match="Full model text"):
            tokenizer.tokenize("Task", state=state, subgoal="a long grounded subgoal")
    with pytest.raises(ValueError, match="Full model text"):
        tokenizer.tokenize("Task", state=np.zeros(8))


def test_grounded_path_needs_no_simple_pair(system):
    system.mode = "grounded"
    spec = {k: v for k, v in SPEC.items() if k != "simple_subgoal"}
    system.handle(request(system, "prompt", subgoal=spec), "grounded")
    assert system.subgoal["simple_subgoal"] is None
    assert system.subgoal["grounded_subgoal"] == "pick up the red cube at <128, 64> for the first time"
    assert not system.runtime.calls
    system.handle(request(system, "release", prefix=1), "grounded-release")
    assert system.policy.count == 2


def test_rejected_token_budget_keeps_old_prompt_and_candidate(system, monkeypatch):
    initialize(system)
    identifier = system.candidate["candidate_id"]

    def reject(_obs):
        raise ValueError("Full model text requires 257 tokens; limit is 256")

    monkeypatch.setattr(system.policy, "validate", reject)
    with pytest.raises(ValueError, match="Full model text"):
        system.handle(request(system, "prompt", subgoal=SPEC), "too-long")
    assert system.candidate["candidate_id"] == identifier
    assert system.version == 1
    assert system.policy.count == 1
    assert not system.runtime.calls


def test_ik_same_frame_still_invalidates_candidate(system):
    initialize(system)
    old = request(system, "release", prefix=1)
    system.runtime.status = "ik"
    system.handle(request(system, "intervene", sequence=[{"target": [0.0] * 7}]), "ik")
    assert system.frame == 0
    assert system.version == 1
    with pytest.raises(ValueError, match="candidate_id"):
        system.handle(old, "old-after-ik")


@pytest.mark.parametrize(
    "sequence", [[], [None], [{"target": [0.0] * 7}] * 17, [{"target": [0.0] * 7, "steps": 100}] * 2]
)
def test_intervention_bounds_checked_before_invalidating(system, sequence):
    initialize(system)
    identifier = system.candidate["candidate_id"]
    with pytest.raises(ValueError, match="interven"):
        system.handle(request(system, "intervene", sequence=sequence), "bad-sequence")
    assert system.candidate["candidate_id"] == identifier
    assert not system.runtime.calls


def test_empty_experience_snapshot(tmp_path):
    manifest = experience.snapshot(tmp_path, tmp_path / "missing.json", enabled=True)
    assert manifest["sha256"]
    assert rgb.read_json(tmp_path / "experience-input.json")["entries"] == []


def test_concurrent_experience_commits_do_not_overwrite(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    source = tmp_path / "experience.json"
    runs = [tmp_path / f"run{i}" for i in range(4)]
    for run in runs:
        run.mkdir()
        rgb.write_json(
            run / "manifest.json",
            {
                "experience_update": True,
                "mode": "grounded",
                "experience_input": {"source": str(source), "enabled": False},
            },
        )
        rgb.write_json(run / "result.json", {"lifecycle": "closed", "recorded_frames": 1})

    def commit(run):
        return experience.commit(
            run,
            {
                "entries": [
                    {"claim": run.name, "conditions": "mock", "confidence": "low", "frames": [0], "kind": "hypothesis"}
                ]
            },
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(commit, runs))
    saved = rgb.read_json(source)
    assert saved["version"] == 4
    assert {e["claim"] for e in saved["entries"]} == {r.name for r in runs}
    assert sorted(r["version"] for r in receipts) == [1, 2, 3, 4]


def test_real_symbolic_transform_selects_correct_field():
    import ast
    import dataclasses

    source = rgb.ROOT / "src/mme_vla_suite/training/config.py"
    node = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "TokenizePromptWithSymbolicMemory"
    )
    scope = {"dataclasses": dataclasses, "DataTransformFn": object, "PaligemmaTokenizer": object, "DataDict": dict}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    for kind in ("simple_subgoal", "grounded_subgoal"):
        calls = []

        def tokenize(prompt, state=None, subgoal=None, _calls=calls):
            _calls.append((prompt, state, subgoal))
            return np.zeros(4), np.ones(4)

        transform = scope["TokenizePromptWithSymbolicMemory"](
            SimpleNamespace(tokenize=tokenize), discrete_state_input=False, symbolic_memory_type=kind
        )
        result = transform({"prompt": "official task", "simple_subgoal": "SIMPLE", "grounded_subgoal": "GROUNDED"})
        assert calls == [
            ("official task", None, None),
            ("official task", None, "SIMPLE" if kind == "simple_subgoal" else "GROUNDED"),
        ]
        assert "symbolic_tokenized_prompt" in result
        assert "simple_subgoal" not in result
        assert "grounded_subgoal" not in result


def test_audit_detects_changed_raw_actions_and_reused_candidate(system):
    from scripts.codex_supervisor.audit import verify_events

    initialize(system)
    system.handle(request(system, "release", prefix=2), "release")
    reset = system.directory / "observations/reset"
    reset.mkdir(parents=True)
    rgb.write_json(reset / "batch.json", {"first_frame": 0, "frame_count": 1})
    events = [json.loads(line) for line in (system.directory / "events.jsonl").read_text().splitlines()]
    verify_events(system.directory, events, system.state)
    raw = system.directory / "internal/controls.jsonl"
    original = raw.read_text()
    controls = [json.loads(line) for line in original.splitlines()]
    controls[0]["action"][0] += 0.1
    raw.write_text("\n".join(json.dumps(c) for c in controls) + "\n")
    with pytest.raises(ValueError, match="original candidate prefix"):
        verify_events(system.directory, events, system.state)
    raw.write_text(original)
    release = next(e for e in events if e["event"] == "release")
    with pytest.raises(ValueError, match="invalidated candidate"):
        verify_events(system.directory, [*events, {**release, "frame": system.frame}], system.state)


def test_nested_run_accepts_inline_input_and_queries(tmp_path, capsys):
    from scripts.codex_supervisor.layout import artifacts
    from scripts.codex_supervisor.layout import create_run

    run = tmp_path / "episode"
    directory = create_run(run)
    assert artifacts(run) == artifacts(directory) == directory
    (directory / "requests").mkdir()
    (directory / "responses").mkdir()
    rgb.write_json(directory / "result.json", {"lifecycle": "ready"})
    rgb.write_json(directory / "responses/finish.json", {"ok": True})
    args = SimpleNamespace(
        run_dir=run,
        client_id="owner",
        request_id="finish",
        request=None,
        request_json='{"operation":"finish"}',
        timeout=1,
    )
    cli.submit(args)
    cli.query(args)
    assert '"ok": true' in capsys.readouterr().out
    assert {p.name for p in run.iterdir()} == {"artifacts", "rollout.mp4", "memory.md", "review.md"}
    entry = json.loads((directory / "inputs.jsonl").read_text())
    assert entry["request"] == {"operation": "finish", "client_id": "owner"}
    (directory / "memory.md").write_text("live memory")
    assert (run / "memory.md").read_text() == "live memory"


def test_video_labels_executed_source_and_prompt_without_changing_rgb(tmp_path, monkeypatch):
    import imageio.v2 as imageio
    from PIL import Image

    from scripts.codex_supervisor import video

    encoded = []
    contexts = []
    original_label = video.labeled_frame

    def label(front, wrist, frame_id, context, task):
        contexts.append((frame_id, dict(context), task))
        return original_label(front, wrist, frame_id, context, task)

    monkeypatch.setattr(video, "labeled_frame", label)
    monkeypatch.setattr(
        imageio,
        "get_writer",
        lambda *a, **k: SimpleNamespace(append_data=lambda frame: encoded.append(frame.copy()), close=lambda: None),
    )
    (tmp_path / "observations").mkdir()
    recorder = rgb.Recorder(tmp_path, 20)
    pixels = np.full((32, 32, 3), 77, dtype=np.uint8)
    obs = {
        "front_rgb_list": [pixels],
        "wrist_rgb_list": [pixels],
        "joint_state_list": [np.zeros(7)],
        "eef_state_list": [np.zeros(6)],
        "gripper_state_list": [np.zeros(2)],
    }
    for index, source in enumerate(("RESET", "VLA", "INTERVENTION")):
        recorder.video_context = {"source": source, "subgoal": f"prompt {index}", "request_id": str(index)}
        recorder.capture(obs, {"task_goal": "official task", "secret": "NOT_PUBLIC"}, f"batch{index}", {})
    recorder.close()
    records = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    assert [r["execution"]["source"] for r in records] == ["RESET", "VLA", "INTERVENTION"]
    assert [context[1]["subgoal"] for context in contexts] == ["prompt 0", "prompt 1", "prompt 2"]
    assert all(context[2] == "official task" for context in contexts)
    assert len(encoded) == 3
    assert all(frame.shape == (768, 1024, 3) for frame in encoded)
    assert tuple(encoded[1][0, 0]) != tuple(encoded[2][0, 0])  # distinct VLA/intervention banners
    for record in records:
        with Image.open(tmp_path / record["rgb"]["front"]) as saved:
            np.testing.assert_array_equal(np.asarray(saved), pixels)
