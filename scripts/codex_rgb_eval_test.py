"""Tests of the observation boundary and result semantics, without a simulator."""

from contextlib import nullcontext
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import codex_rgb_eval as bridge


@pytest.mark.parametrize(
    "target",
    [
        [0] * 6,
        [0] * 8,
        [0, 0, 0, 0, 0, 0, 2],
        [True, 0, 0, 0, 0, 0, 1],
        ["0", 0, 0, 0, 0, 0, 1],
        [float("nan"), 0, 0, 0, 0, 0, 1],
        [float("inf"), 0, 0, 0, 0, 0, 1],
        {"object": "cube"},
    ],
)
def test_reject_non_numeric_or_invalid_actions(target):
    with pytest.raises(ValueError, match="target|gripper"):
        bridge.validate_action({"target": target})


@pytest.mark.parametrize(("steps", "interpolate"), [(0, 0), (101, 0), (True, 0), (2, 3), (2, -1), (2, 0.5)])
def test_reject_invalid_duration(steps, interpolate):
    with pytest.raises(ValueError, match="steps|interpolate"):
        bridge.validate_action({"target": [0] * 7, "steps": steps, "interpolate": interpolate})


def test_short_angle_interpolation():
    start = [0, 0, 0, 0, 0, math.pi - 0.1]
    target = [0.2, 0, 0.2, 0, 0, -math.pi + 0.1, -1]
    halfway = bridge.action_at(start, target, 0, 2)
    assert halfway == pytest.approx([0.1, 0, 0.1, 0, 0, math.pi, -1])


@pytest.mark.parametrize(
    ("info", "expected"),
    [
        ({}, "incomplete"),
        ({"status": "ongoing"}, "incomplete"),
        ({"status": "success"}, "success"),
        ({"success": np.bool_(1)}, "success"),
        ({"status": "fail"}, "failure"),
        ({"status": "timeout"}, "timeout"),
        ({"status": "error", "success": True}, "error"),
    ],
)
def test_outcome_uses_official_feedback(info, expected):
    assert bridge.outcome(bridge.safe_feedback(info)) == expected


def test_only_structured_ik_failure_is_recoverable():
    info = {
        "status": "error",
        "error_message": "ee step (rpy): IK failed (status=IK Failed! Distance 0.008735 is greater than threshold 0.001000., num_solutions=0)",
    }
    assert bridge.recoverable_control_error(info) == {
        "type": "ik_failed",
        "retryable": True,
        "num_solutions": 0,
        "distance": 0.008735,
        "threshold": 0.001,
    }
    assert bridge.recoverable_control_error({"status": "error", "error_message": "SECRET"}) is None


def test_filter_and_preserve_every_frame(tmp_path, monkeypatch):
    import imageio.v2 as imageio

    video_frames = []

    class Writer:
        def append_data(self, frame):
            video_frames.append(frame)

        def close(self):
            pass

    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: Writer())
    (tmp_path / "observations").mkdir()
    frames = [np.full((32, 32, 3), i, dtype=np.uint8) for i in range(14)]
    obs = {
        "front_rgb_list": frames,
        "wrist_rgb_list": frames,
        "joint_state_list": [np.zeros(7)] * 14,
        "eef_state_list": [np.arange(6)] * 14,
        "gripper_state_list": [np.zeros(2)] * 14,
        "front_depth_list": "SECRET_DEPTH",
        "maniskill_obs": "SECRET_STATE",
    }
    info = {
        "task_goal": "Remember the demonstrated order",
        "success": False,
        "grounded_subgoal_online": "SECRET_ANSWER",
        "simple_subgoal_online": "SECRET_LABEL",
        "error_message": "SECRET_ERROR",
        "reward": 0.998,
    }
    recorder = bridge.Recorder(tmp_path, 20)
    recorder.capture(obs, info, "reset", bridge.safe_feedback(info))
    recorder.close()
    assert recorder.frame_count == len(video_frames) == 14
    assert len(list(tmp_path.rglob("*.png"))) == 28
    assert len(list(tmp_path.rglob("sheet_*.jpg"))) == 2
    records = [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]
    assert [record["frame_id"] for record in records] == list(range(14))
    latest = bridge.read_json(tmp_path / "latest.json")
    assert latest["frame_id"] == 13
    assert latest["robot"]["eef_state"] == list(range(6))
    assert latest["task_goal"] == info["task_goal"]
    public_text = "".join(path.read_text() for path in tmp_path.rglob("*.json*"))
    assert "SECRET" not in public_text
    assert "0.998" not in public_text


def test_empty_error_preserves_previous_observation(tmp_path, monkeypatch):
    import imageio.v2 as imageio

    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: None)
    (tmp_path / "observations").mkdir()
    recorder = bridge.Recorder(tmp_path, 20)
    recorder.latest = {"frame_id": 5, "task_goal": "Task"}
    recorder.capture(None, {"status": "error"}, "error", bridge.safe_feedback({"status": "error"}))
    assert recorder.latest["frame_id"] == 5
    assert recorder.latest["task_goal"] == "Task"
    assert recorder.latest["feedback"]["execution_error"]


@pytest.mark.parametrize(
    "status", ["ongoing", "success", "fail", "timeout", "error", "ik_error", "exception", "budget"]
)
def test_server_protocol_and_finalization(tmp_path, monkeypatch, status):
    import imageio.v2 as imageio

    directory = tmp_path / "run"
    closed = []
    obs = {
        "front_rgb_list": [np.zeros((32, 32, 3), dtype=np.uint8)],
        "wrist_rgb_list": [np.zeros((32, 32, 3), dtype=np.uint8)],
        "joint_state_list": [np.zeros(7)],
        "eef_state_list": [np.zeros(6)],
        "gripper_state_list": [np.zeros(2)],
    }

    class Writer:
        def append_data(self, frame):
            pass

        def close(self):
            closed.append("video")

    class Environment:
        def reset(self):
            bridge.write_json(
                directory / "requests/000.json",
                {
                    "operation": "act",
                    "target": [0] * 7,
                    "expected_frame": -1,
                },
            )
            bridge.write_json(
                directory / "requests/001.json",
                {
                    "operation": "act",
                    "target": [0] * 7,
                    "steps": 5,
                    "expected_frame": 0,
                },
            )
            return obs, {"task_goal": "Task", "grounded_subgoal_online": "SECRET"}

        def step(self, action):
            assert action == [0] * 7
            if status == "exception":
                raise RuntimeError("SECRET")
            if status in ("ongoing", "ik_error"):
                bridge.write_json(directory / "requests/002.json", {"operation": "finish"})
            if status == "ik_error":
                return {}, 0.0, True, False, {
                    "status": "error",
                    "error_message": "ee step (rpy): IK failed (status=IK Failed! Distance 0.008735 is greater than threshold 0.001000., num_solutions=0)",
                }
            return obs, 0.987654321, False, False, {"status": status, "error_message": "SECRET"}

        def close(self):
            closed.append("env")

    monkeypatch.setattr(bridge, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(bridge, "quiet_simulator", nullcontext)
    monkeypatch.setattr(bridge.signal, "signal", lambda *args: None)
    monkeypatch.setattr(bridge, "build_environment", lambda args: (Environment(), {"seed": 7}))
    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: Writer())
    args = SimpleNamespace(
        run_dir=directory,
        task="PickXtimes",
        dataset="test",
        episode=0,
        seed=None,
        max_steps=3 if status == "budget" else 10,
        fps=20,
        sim_backend="cpu",
        render_backend=None,
        experience=None,
    )
    if status in ("error", "exception"):
        with pytest.raises(SystemExit, match="1"):
            bridge.serve(args)
    else:
        bridge.serve(args)
    result = bridge.read_json(directory / "result.json")
    assert result["lifecycle"] == "closed"
    assert (
        result["outcome"]
        == {
            "ongoing": "incomplete",
            "success": "success",
            "fail": "failure",
            "timeout": "timeout",
            "error": "error",
            "ik_error": "incomplete",
            "exception": "error",
            "budget": "incomplete",
        }[status]
    )
    assert result["video_finalized"]
    assert closed == ["video", "env"]
    assert result["action_requests"] == 1
    assert result["control_steps"] == {"ongoing": 5, "budget": 3}.get(status, 1)
    if status == "ik_error":
        assert result["last_control_error"]["type"] == "ik_failed"
        response = bridge.read_json(directory / "responses/001.json")
        assert response["ok"]
        assert response["control_error"]["distance"] == pytest.approx(0.008735)
    if status == "budget":
        assert result["stop_reason"] == "control_budget"
    assert not bridge.read_json(directory / "responses/000.json")["ok"]
    assert (directory / "responses/001.json").exists()
    text = "".join(path.read_text() for path in directory.rglob("*.json*"))
    assert "SECRET" not in text
    assert "0.987654321" not in text
