"""Bounded delegation and failure recovery against the real file bridge and a deterministic robot."""

import base64
import hashlib
import io
import time

from PIL import Image
import pytest

from scripts import codex_rgb_eval as rgb
from scripts.robomme_agent import jev
from scripts.robomme_agent import metrics
from scripts.robomme_agent import supervision
from scripts.robomme_agent.tools import EpisodeTools
from scripts.robomme_agent_test import episode  # noqa: F401
from scripts.robomme_agent_test import identity
from scripts.robomme_agent_test import initialize


@pytest.mark.parametrize(
    ("grounded", "visual"),
    [
        ("pick up the third green cube at <81, 118>", "pick up the third green cube"),
        ("pick up the peg by grasping the far end at <0, 255>", "pick up the peg by grasping the far end"),
        ("hook the cube at <12,34> to the target at <56, 78> with the peg", "hook the cube to the target with the peg"),
        ("pick up the container at <12, 34> that hides the red cube", "pick up the container that hides the red cube"),
        ("move forward", "move forward"),
        ("pick up the 2nd cube at <9, 10>", "pick up the 2nd cube"),
    ],
)
def test_default_completion_question_removes_only_grounding(grounded, visual):
    q = supervision.default_completion_question(grounded)
    assert q["prompt"] == (
        "In these current front and wrist views, is this visible subgoal already accomplished? "
        + visual
        + " Answer uncertain if the evidence is insufficient; an attempted action is not completion."
    )
    jev.validate_question(q)


@pytest.fixture
def delegated(episode, monkeypatch):  # noqa: F811
    episode.runtime.robot["gripper_state"] = [0.02, 0.02]
    episode.recorder.capture({"robot": episode.runtime.robot}, {}, "reset", rgb.safe_feedback({}))
    episode.supervisor.publish()

    def describe(actions, current):
        return {"trajectory": [{"target_xyz_rpy": current, "gripper_command": float(row[7])} for row in actions]}

    episode.supervisor.fk.describe = describe
    tools = EpisodeTools(episode.directory, timeout=0.001, supervision_mode="rules-only")
    initialize(tools)

    def panel(_tools, _frame, target):
        Image.new("RGB", (512, 280), "red").save(target)
        return {"front": "front-hash", "wrist": "wrist-hash"}

    monkeypatch.setattr(supervision, "current_panel", panel)
    return tools, episode


def contract(tools, **changes):
    return {
        **identity(tools),
        "phase": "approach",
        "phase_mode": "ordinary",
        "max_chunks": 3,
        "max_no_progress_checks": 3,
        "motion_expected": False,
        "grasp_state": "none",
        "wake_on_gripper_change": True,
        "questions": [
            {
                "prompt": "Has the target been reached?",
                "purpose": "stage_warning",
                "labels": ["near", "far", "uncertain"],
                "review_labels": ["near", "uncertain"],
            }
        ],
        "experience_refs": [],
        "reason": "approach under supervision",
        **changes,
    }


def enable_jev(tools, monkeypatch, label=1):
    config = {
        "protocol": "omnijev-mso1-v1",
        "endpoint": "http://localhost:1/v1/decide",
        "model": "test",
        "revision": "test",
        "timeout_seconds": 0.02,
        "max_calls": 30,
        "max_total_seconds": 1,
        "probability_policy": {"min_probability": 0.6, "min_margin": 0.1, "max_abstain": 0.2},
        "questions": {"q": contract(tools)["questions"][0]},
    }
    path = tools.directory / "test-profile.json"
    rgb.write_json(path, config)
    tools.jev = jev.VisualAdviser(path)
    tools.supervision_mode = "jev"
    calls = []

    def post(_endpoint, payload, _headers, _timeout):
        calls.append(payload)
        probabilities = [0.01, 0.01, 0.01]
        probabilities[label] = 0.96
        return {
            "options": payload["options"],
            "choice_index": label,
            "probabilities": probabilities,
            "abstain_probability": 0.02,
            "model": "test",
        }

    monkeypatch.setattr(jev, "post", post)
    return calls


def test_delegation_uses_fresh_candidates_and_does_not_ack_astra(delegated):
    tools, harness = delegated
    args = contract(tools)
    seen = list(tools.state["seen"])
    result, _ = tools.execute("supervise_vla", args, "delegate")
    assert result["executed_chunks"] == 3
    assert result["executed_steps"] == len(harness.runtime.calls) == 6
    assert set(result["wake_reasons"]) == {"chunk_budget"}
    assert tools.state["seen"] == seen
    with pytest.raises(ValueError, match="history"):
        tools.execute("release", {**identity(tools), "prefix": 1, "reason": "unseen"}, "illegal")
    tools.execute("supervise_vla", args, "delegate")
    assert len(harness.runtime.calls) == 6
    bundle, images = tools.handoff("wake")
    assert bundle["delivered_history"]
    assert images
    tools.acknowledge_images()
    assert tools.status()["unseen_count"] == 0


def test_pending_child_recovers_once_after_process_restart(delegated):
    tools, harness = delegated
    harness.automatic = False
    args = contract(tools)
    result, _ = tools.execute("supervise_vla", args, "delegate")
    assert result["pending"]
    original = result["request_id"]
    assert not harness.runtime.calls
    harness.process(original)
    harness.automatic = True
    restarted = EpisodeTools(harness.directory, timeout=0.001, supervision_mode="rules-only")
    result, _ = restarted.execute("query", {"request_id": original}, "resolve")
    assert result["executed_steps"] == 6
    assert len(harness.runtime.calls) == 6
    restarted.execute("query", {"request_id": original}, "resolve")
    assert len(harness.runtime.calls) == 6


def test_jev_can_monitor_multiple_chunks_without_astra(delegated, monkeypatch):
    tools, harness = delegated
    calls = enable_jev(tools, monkeypatch)
    result, _ = tools.execute("supervise_vla", contract(tools), "delegate")
    assert len(calls) == 2
    assert result["executed_steps"] == 6
    assert tools.state["jev_accounting"]["calls"] == 2
    assert tools.status()["unseen_count"] == 6
    assert metrics.summarize(tools)["delegated_steps"] == 6


def test_astra_authorizes_first_chunk_then_alert_prevents_next_execution(delegated, monkeypatch):
    tools, harness = delegated
    calls = enable_jev(tools, monkeypatch, label=0)
    result, _ = tools.execute("supervise_vla", contract(tools), "delegate")
    assert result["wake_reasons"] == ["jev_alert"]
    assert len(harness.runtime.calls) == result["executed_steps"] == 2
    assert len(calls) == 1
    assert tools.state["delegations"][result["contract_id"]]["initial_review"] == "astra"


def test_default_question_sent_without_changing_vla_grounding(delegated, monkeypatch):
    tools, _ = delegated
    calls = enable_jev(tools, monkeypatch)
    grounded = tools.status()["decision"]["subgoal"]["grounded_subgoal"]
    assert "<" in grounded
    args = contract(tools)
    del args["questions"]
    result, _ = tools.execute("supervise_vla", args, "default-question")
    assert result["wake_reasons"] == ["chunk_budget"]
    assert calls
    assert all("<" not in c["question"] for c in calls)
    assert all(c["question"] == supervision.default_completion_question(grounded)["prompt"] for c in calls)
    assert tools.status()["decision"]["subgoal"]["grounded_subgoal"] == grounded


@pytest.mark.parametrize(("camera", "color"), [("front", (255, 0, 0)), ("wrist", (0, 0, 255))])
def test_question_camera_delivers_only_selected_pixels_and_records_sources(delegated, monkeypatch, camera, color):
    tools, _ = delegated
    calls = enable_jev(tools, monkeypatch)

    def panel(_tools, _frame, target):
        image = Image.new("RGB", (512, 280), "red")
        image.paste(Image.new("RGB", (256, 280), "blue"), (256, 0))
        image.save(target)
        return {"front": "front-hash", "wrist": "wrist-hash"}

    monkeypatch.setattr(supervision, "current_panel", panel)
    args = contract(tools)
    args["questions"][0]["camera"] = camera
    result, _ = tools.execute("supervise_vla", args, "single-camera")
    assert result["executed_chunks"] == 3
    assert len(calls) == 2
    for payload, advice in zip(calls, result["advice"], strict=True):
        pixels = base64.b64decode(payload["state"][0]["image"].split(",", 1)[1])
        with Image.open(io.BytesIO(pixels)) as image:
            assert image.size == (256, 280)
            assert image.getpixel((128, 140)) == color
        assert advice["camera"] == camera
        assert advice["source_sha256"] == {camera: f"{camera}-hash"}
        assert advice["image_sha256"] == hashlib.sha256(pixels).hexdigest()
    tools.execute("supervise_vla", args, "single-camera")
    assert len(calls) == 2  # Replay cannot issue observations or physical controls again.
    assert tools.status()["unseen_count"] == 6  # JEV delivery is not Astra review.


def test_mandatory_gripper_transition_bypasses_model(delegated, monkeypatch):
    tools, harness = delegated
    calls = enable_jev(tools, monkeypatch)
    harness.supervisor.candidate["eef"]["trajectory"][0]["gripper_command"] = -1
    harness.supervisor.publish()
    result, _ = tools.execute("supervise_vla", contract(tools), "delegate")
    assert result["wake_reasons"] == ["candidate_gripper_transition"]
    assert not calls
    assert not harness.runtime.calls


def test_hard_timeout_disables_delegation_and_late_result_cannot_move(delegated, monkeypatch):
    tools, harness = delegated
    enable_jev(tools, monkeypatch)

    def slow(*_):
        time.sleep(0.1)
        return {}

    monkeypatch.setattr(jev, "post", slow)
    result, _ = tools.execute("supervise_vla", contract(tools), "delegate")
    assert result["wake_reasons"] == ["jev_unavailable"]
    assert tools.state["jev_disabled"]
    assert len(harness.runtime.calls) == 2
    with pytest.raises(ValueError, match="unavailable"):
        tools.execute("supervise_vla", contract(tools), "retry")


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_id": "stale"},
        {"subgoal_version": 999},
        {"max_chunks": 11},
        {"checkpoint_steps": 999},
        {"prefix_steps": 1},
        {"phase_mode": "critical"},
        {"phase_mode": "timing"},
    ],
)
def test_invalid_contract_rejected_without_movement(delegated, changes):
    tools, harness = delegated
    with pytest.raises(ValueError, match="stale|exceeds|require|only|unsupported"):
        tools.execute("supervise_vla", contract(tools, **changes), "invalid")
    assert not harness.runtime.calls


def test_no_progress_and_partial_official_end(delegated):
    tools, harness = delegated
    result, _ = tools.execute(
        "supervise_vla", contract(tools, motion_expected=True, max_no_progress_checks=1), "delegate"
    )
    assert result["wake_reasons"] == ["no_robot_progress"]
    assert len(harness.runtime.calls) == 2
    tools.handoff("wake")
    tools.acknowledge_images()
    harness.supervisor.max_steps = 3
    result, _ = tools.execute("supervise_vla", contract(tools), "next")
    assert result["wake_reasons"] == ["episode_ended"]
    assert result["executed_steps"] == 1
    assert len(harness.runtime.calls) == 3


def test_low_confidence_and_abstention_trigger_review(delegated, monkeypatch):
    tools, harness = delegated
    enable_jev(tools, monkeypatch)
    monkeypatch.setattr(
        jev,
        "post",
        lambda *args: {
            "options": args[1]["options"],
            "choice_index": 1,
            "probabilities": [0.1, 0.5, 0.1],
            "abstain_probability": 0.3,
        },
    )
    result, _ = tools.execute("supervise_vla", contract(tools), "delegate")
    assert result["wake_reasons"] == ["jev_low_confidence"]
    assert len(harness.runtime.calls) == 2


def test_finish_settles_pending_child_without_releasing_remaining_budget(delegated):
    tools, harness = delegated
    harness.automatic = False
    result, _ = tools.execute("supervise_vla", contract(tools), "delegate")
    harness.process(result["request_id"])
    harness.automatic = True
    tools.finish("operator stop")
    assert len(harness.runtime.calls) == 2
    saved = next(iter(tools.state["delegations"].values()))
    assert saved["steps"] == 2
    assert saved["wake_reasons"] == ["delegation_cancelled"]
    assert tools.status()["result"]["lifecycle"] == "closed"


def test_changed_candidate_during_advice_is_rejected(delegated, monkeypatch):
    tools, harness = delegated
    enable_jev(tools, monkeypatch)
    normal = jev.post

    def changed(*args):
        result = normal(*args)
        harness.supervisor.infer()
        return result

    monkeypatch.setattr(jev, "post", changed)
    result, _ = tools.execute("supervise_vla", contract(tools), "delegate")
    assert result["wake_reasons"] == ["stale_advice"]
    assert len(harness.runtime.calls) == 2


def test_missing_visual_input_falls_back_without_stuck_contract(delegated, monkeypatch):
    tools, harness = delegated
    enable_jev(tools, monkeypatch)

    def missing(*_):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(supervision, "current_panel", missing)
    result, _ = tools.execute("supervise_vla", contract(tools), "delegate")
    assert result["wake_reasons"] == ["invalid_visual_input"]
    assert tools.state["active_delegation"] is None
    assert len(harness.runtime.calls) == 2


def test_single_chunk_contract_needs_no_jev_query(delegated, monkeypatch):
    tools, harness = delegated
    calls = enable_jev(tools, monkeypatch)
    result, _ = tools.execute("supervise_vla", contract(tools, max_chunks=1), "delegate")
    assert result["executed_steps"] == len(harness.runtime.calls) == 2
    assert not calls
    assert set(result["wake_reasons"]) == {"chunk_budget"}


def test_delegation_schema_exposes_only_chunk_budget():
    assert "max_chunks" in supervision.CONTRACT_PROPERTIES
    assert "checkpoint_steps" not in supervision.CONTRACT_PROPERTIES
    assert "prefix_steps" not in supervision.CONTRACT_PROPERTIES


def test_legacy_pending_contract_keeps_original_step_checkpoint(delegated):
    tools, harness = delegated
    harness.automatic = False
    result, _ = tools.execute("supervise_vla", contract(tools), "delegate")
    saved = tools.state["delegations"][result["contract_id"]]
    saved["args"].update(prefix_steps=2, checkpoint_steps=2)
    saved.pop("chunk_prefix_limit")
    tools.save()
    harness.process(result["request_id"])
    harness.automatic = True
    restarted = EpisodeTools(harness.directory, timeout=0.001, supervision_mode="rules-only")
    result, _ = restarted.execute("query", {"request_id": result["request_id"]}, "resolve")
    assert result["executed_steps"] == len(harness.runtime.calls) == 2
    assert result["wake_reasons"] == ["phase_checkpoint"]


@pytest.mark.parametrize(("mode", "prefix"), [("ordinary", 16), ("timing", 4)])
def test_chunk_prefix_is_selected_by_harness_not_jev(delegated, monkeypatch, mode, prefix):
    tools, harness = delegated
    harness.supervisor.prefix = 16
    harness.supervisor.max_steps = 100
    harness.supervisor.infer()
    calls = enable_jev(tools, monkeypatch)
    result, _ = tools.execute("supervise_vla", contract(tools, phase_mode=mode, max_chunks=1), "delegate")
    assert result["executed_chunks"] == 1
    assert result["executed_steps"] == len(harness.runtime.calls) == prefix
    assert result["wake_reasons"] == ["chunk_budget"]
    assert not calls


def test_sparse_observation_does_not_change_chunk_budget_or_image_ownership(delegated, monkeypatch):
    tools, harness = delegated
    calls = enable_jev(tools, monkeypatch)
    tools.jev.config["reporting_policy"] = {"ordinary_observation_interval_chunks": 2}
    seen = list(tools.state["seen"])
    args = contract(tools)
    result, _ = tools.execute("supervise_vla", args, "delegate")
    assert result["executed_steps"] == len(harness.runtime.calls) == 6
    assert result["executed_chunks"] == 3
    assert len(calls) == 1
    assert result["wake_reasons"] == ["chunk_budget"]
    assert tools.state["seen"] == seen
    saved = tools.state["delegations"][result["contract_id"]]
    assert saved["observation_interval_chunks"] == 2
    tools.execute("supervise_vla", args, "delegate")
    assert len(calls) == 1
    assert len(harness.runtime.calls) == 6


def test_sparse_observation_still_returns_on_low_confidence(delegated, monkeypatch):
    tools, harness = delegated
    enable_jev(tools, monkeypatch)
    tools.jev.config["reporting_policy"] = {"ordinary_observation_interval_chunks": 2}
    monkeypatch.setattr(
        jev,
        "post",
        lambda *args: {
            "options": args[1]["options"],
            "choice_index": 1,
            "probabilities": [0.1, 0.5, 0.1],
            "abstain_probability": 0.3,
            "model": "test",
        },
    )
    result, _ = tools.execute("supervise_vla", contract(tools), "delegate")
    assert result["wake_reasons"] == ["jev_low_confidence"]
    assert result["executed_steps"] == len(harness.runtime.calls) == 4


def test_hard_rule_runs_between_sparse_model_observations(delegated, monkeypatch):
    tools, harness = delegated
    calls = enable_jev(tools, monkeypatch)
    tools.jev.config["reporting_policy"] = {"ordinary_observation_interval_chunks": 2}
    original = harness.supervisor.infer

    def changed():
        original()
        if harness.runtime.calls:
            harness.supervisor.candidate["eef"]["trajectory"][0]["gripper_command"] = -1
            harness.supervisor.publish()

    monkeypatch.setattr(harness.supervisor, "infer", changed)
    result, _ = tools.execute("supervise_vla", contract(tools), "delegate")
    assert result["executed_steps"] == len(harness.runtime.calls) == 2
    assert result["wake_reasons"] == ["candidate_gripper_transition"]
    assert not calls
