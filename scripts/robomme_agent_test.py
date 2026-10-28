"""API-supervisor contracts using a deterministic in-memory robot, never a real API/GPU."""

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import codex_rgb_eval as rgb
from scripts import codex_symbolic_eval as bridge
from scripts.codex_supervisor import experience
from scripts.codex_supervisor.core import Supervisor
from scripts.codex_supervisor.mock_policy import StallPolicy
from scripts.robomme_agent import runner
from scripts.robomme_agent.tools import EpisodeTools
from scripts.run_robomme_agent import apply_pending_launch_settings
from scripts.run_robomme_agent import parser

SPEC = {"text": "press the button at <point>", "point_yx_1000": [[500, 500]]}


@pytest.mark.parametrize("resume", [None, "attach", "resume_review"])
def test_pending_launch_budget_only_changes_fresh_episode(tmp_path, resume):
    args = parser().parse_args(["--run-dir", str(tmp_path / "ep1"), "--max-steps", "1000"])
    if resume:
        setattr(args, resume, tmp_path / "ep1")
    (tmp_path / "pending-launch-settings.json").write_text(json.dumps({"max_steps": 800, "tasks": [args.task]}))
    apply_pending_launch_settings(args)
    assert args.max_steps == (1000 if resume else 800)


def test_pending_launch_budget_respects_task_scope(tmp_path):
    args = parser().parse_args(["--run-dir", str(tmp_path / "ep1"), "--max-steps", "1000"])
    (tmp_path / "pending-launch-settings.json").write_text(json.dumps({"max_steps": 800, "tasks": ["StopCube"]}))
    apply_pending_launch_settings(args)
    assert args.max_steps == 1000


@pytest.fixture
def episode(tmp_path, monkeypatch):
    for name in ("requests", "responses", "internal", "observations"):
        (tmp_path / name).mkdir()
    for name in ("memory.md", "review.md"):
        (tmp_path / name).write_text("initial")
    rgb.write_json(
        tmp_path / "manifest.json",
        {
            "task": "PickHighlight",
            "mode": "grounded",
            "policy_backend": "mock-stall",
            "client_id": "test",
            "experience_input": {"enabled": False, "source": str(tmp_path / "engineering.json")},
            "experience_update": True,
        },
    )

    class Runtime:
        def __init__(self):
            self.calls = []
            self.robot = {"eef_state": [0.0, 0.0, 0.5, 0.0, 0.0, 0.0], "joint_state": [0.0] * 7, "gripper_state": [1.0]}
            self.ik_error = False

        def image_hw(self):
            return (16, 32)

        def observation(self, subgoal):
            return {
                "observation/state": np.array(self.robot["joint_state"] + [1.0]),
                "prompt": "Official task",
                **subgoal,
            }

        def step(self, action, kind):
            self.calls.append((kind, action))
            if self.ik_error:
                return (
                    {},
                    0,
                    True,
                    False,
                    {
                        "status": "error",
                        "error_message": "ee step (rpy): IK failed (status=IK Failed! Distance 0.01 is greater than threshold 0.001., num_solutions=0)",
                    },
                )
            self.robot = {**self.robot, "eef_state": action[:6] if kind == "eef" else self.robot["eef_state"]}
            return {"robot": self.robot}, 0, False, False, {"task_goal": "Official task", "hidden": "SECRET"}

    class Recorder:
        def __init__(self):
            self.frame_count = 0
            self.video_context = None
            self.latest = {}

        def capture(self, obs, info, batch, feedback):
            if not obs:
                return
            index = self.frame_count
            images = {}
            for camera in ("front", "wrist"):
                relative = f"observations/{index}-{camera}.png"
                Image.new("RGB", (32, 16), (index, 50, 100)).save(tmp_path / relative)
                images[camera] = relative
            self.latest = {
                "frame_id": index,
                "robot": obs["robot"],
                "rgb": images,
                "task_goal": "Official task",
                "feedback": feedback,
            }
            rgb.append_json(tmp_path / "frames.jsonl", self.latest)
            rgb.write_json(tmp_path / "latest.json", self.latest)
            self.frame_count += 1

    class FK:
        def describe(self, actions, current):
            return {"trajectory": [{"target_xyz_rpy": current}] * len(actions)}

    runtime, recorder = Runtime(), Recorder()
    recorder.capture({"robot": runtime.robot}, {}, "reset", rgb.safe_feedback({}))
    supervisor = Supervisor(
        tmp_path, recorder, runtime, StallPolicy(), FK(), task="PickHighlight", mode="grounded", max_steps=10, prefix=2
    )
    original = bridge.immutable_request
    harness = SimpleNamespace(
        directory=tmp_path, supervisor=supervisor, runtime=runtime, recorder=recorder, automatic=True
    )

    def process(identifier):
        path = tmp_path / "requests" / f"{identifier}.json"
        response = tmp_path / "responses" / path.name
        if response.exists():
            return
        request = rgb.read_json(path)
        rgb.append_json(
            tmp_path / "events.jsonl",
            {
                "event": "request_started",
                "request_id": identifier,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
        )
        try:
            result = supervisor.handle(request, identifier)
        except ValueError as exc:
            result = {"ok": False, "error": str(exc)}
        if supervisor.closed:
            supervisor.state.update(lifecycle="closed", video_finalized=True)
        supervisor.publish()
        rgb.write_json(response, {**result, "result": supervisor.state, "request_id": identifier})

    def enqueue(directory, identifier, request):
        result = original(directory, identifier, request)
        if harness.automatic:
            process(identifier)
        return result

    harness.process = process
    monkeypatch.setattr(bridge, "immutable_request", enqueue)
    return harness


def tools_for(episode):
    return EpisodeTools(episode.directory, timeout=0.005)


def see_current(tools):
    frame = tools.status()["latest"]["frame_id"]
    tools.execute("history", {"first": 0, "last": frame}, f"history-{frame}")
    tools.execute("view_image", {"frame": frame, "camera": "front"}, f"front-{frame}")
    tools.acknowledge_images()


def identity(tools):
    d = tools.status()["decision"]
    return {
        "expected_frame": d["expected_frame"],
        "subgoal_version": d["subgoal_version"],
        "candidate_id": (d["candidate"] or {}).get("candidate_id"),
    }


def initialize(tools):
    see_current(tools)
    return tools.execute("set_subgoal", {**identity(tools), "subgoal": SPEC, "reason": "visible button"}, "prompt")


def intervention(tools):
    return {
        **identity(tools),
        "sequence": [{"target": [0.0, 0.0, 0.51, 0.0, 0.0, 0.0, 1.0], "steps": 1, "interpolate": 1}],
        "reason": "stationary candidate cannot approach visible button",
    }


def test_mock_holds_measured_joints_without_mutating_state():
    state = np.arange(8, dtype=float) / 10
    before = state.copy()
    policy = StallPolicy()
    result = np.asarray(policy.infer({"observation/state": state})["actions"])
    assert result.shape == (32, 8)
    np.testing.assert_array_equal(result[:, :7], np.tile(state[:7], (32, 1)))
    np.testing.assert_array_equal(state, before)
    assert np.all(result[:, 7] == 1)
    assert policy.metadata["real_vla"] is False
    with pytest.raises(ValueError, match="finite"):
        policy.validate({"observation/state": [float("nan")] * 8})


def test_images_must_be_delivered_and_boundaries_enforced(episode):
    tools = tools_for(episode)
    result, images = tools.execute("history", {"first": 0, "last": 0}, "history")
    assert result["frame_count"] == len(images) == 1
    assert tools.status()["unseen_count"] == 1
    with pytest.raises(ValueError, match="history"):
        tools.execute("set_subgoal", {**identity(tools), "subgoal": SPEC, "reason": "test"}, "too-early")
    tools.acknowledge_images()
    with pytest.raises(ValueError, match="ORIGINAL"):
        tools.execute("set_subgoal", {**identity(tools), "subgoal": SPEC, "reason": "test"}, "no-original")
    with pytest.raises(ValueError, match="unknown tool"):
        tools.execute("shell", {"cmd": "read hidden"}, "forbidden")
    with pytest.raises(ValueError, match="choices"):
        tools.execute("view_image", {"frame": 0, "camera": "../../internal"}, "path")
    with pytest.raises(ValueError, match="frame does not exist"):
        tools.execute("view_image", {"frame": 999, "camera": "front"}, "future")
    assert "SECRET" not in json.dumps(tools.status())


def test_release_stale_rejection_and_duplicate_recovery(episode):
    tools = tools_for(episode)
    initialize(tools)
    args = {**identity(tools), "prefix": 2, "reason": "test original prefix"}
    result, _ = tools.execute("release", args, "release")
    assert result["executed_steps"] == len(episode.runtime.calls) == 2
    new_candidate = tools.status()["decision"]["candidate"]["candidate_id"]
    assert new_candidate != args["candidate_id"]
    tools.execute("release", args, "release")  # cache replay
    assert len(episode.runtime.calls) == 2
    # Crash window: the server replied but the tool-result cache wasn't committed.
    cache = tools.agent_dir / ("api-" + hashlib.sha256(b"release").hexdigest()[:32] + ".json")
    cache.unlink()
    recovered = tools_for(episode)
    recovered.execute("release", args, "release")
    assert len(episode.runtime.calls) == 2
    see_current(recovered)
    result, _ = recovered.execute("release", args, "stale-new-id")
    assert not result["ok"]
    assert "stale" in result["error"]
    assert len(episode.runtime.calls) == 2


def test_pending_action_only_queries_original_request(episode):
    tools = tools_for(episode)
    initialize(tools)
    episode.automatic = False
    args = {**identity(tools), "prefix": 1, "reason": "test timeout"}
    result, _ = tools.execute("release", args, "pending")
    assert result["pending"]
    assert not episode.runtime.calls
    with pytest.raises(ValueError, match="original pending"):
        tools.execute("release", args, "another")
    episode.process(result["request_id"])
    response, _ = tools.execute("query", {"request_id": result["request_id"]}, "query")
    assert response["ok"]
    assert tools.state["pending"] is None
    assert len(episode.runtime.calls) == 1


@pytest.mark.parametrize("ik_error", [False, True])
def test_takeover_requires_real_motion_images_and_new_candidate_review(episode, ik_error):
    tools = tools_for(episode)
    initialize(tools)
    episode.runtime.ik_error = ik_error
    result, _ = tools.execute("intervene", intervention(tools), "intervene")
    assert result["ok"]
    assert not tools.state["reviewed_intervention"]
    see_current(tools)
    frame = tools.status()["latest"]["frame_id"]
    candidate = identity(tools)["candidate_id"]
    tools.execute(
        "write_memory",
        {
            "text": "Reviewed returned observations and fresh stationary candidate.",
            "frames": [frame],
            "candidate_id": candidate,
        },
        "post-review",
    )
    assert tools.state["reviewed_intervention"] is (not ik_error)
    assert tools.state["interventions"][0]["robot_state_changed"] is (not ik_error)


def test_motionless_intervention_does_not_pass(episode):
    tools = tools_for(episode)
    initialize(tools)
    action = intervention(tools)
    action["sequence"][0]["target"][2] = 0.5
    tools.execute("intervene", action, "noop")
    see_current(tools)
    tools.execute(
        "write_memory", {"text": "No movement.", "frames": [1], "candidate_id": identity(tools)["candidate_id"]}, "note"
    )
    assert not tools.state["reviewed_intervention"]


@pytest.mark.parametrize("value", [True, -1, float("nan"), "1"])
def test_invalid_control_arguments_rejected_before_enqueue(episode, value):
    tools = tools_for(episode)
    initialize(tools)
    with pytest.raises(ValueError, match="tool"):
        tools.execute("release", {**identity(tools), "prefix": value, "reason": "invalid"}, "bad")
    assert not episode.runtime.calls


def test_mock_experience_cannot_pollute_task_evidence(episode):
    tools = tools_for(episode)
    see_current(tools)
    tools.finish("engineering check")
    entry = {
        "claim": "Protocol exercised",
        "conditions": "Mock harness",
        "confidence": "high",
        "frames": [0],
        "kind": "observation",
    }
    with pytest.raises(ValueError, match="engineering"):
        experience.commit(episode.directory, {"entries": [entry]})
    receipt = experience.commit(episode.directory, {"entries": [{**entry, "kind": "engineering"}]})
    assert receipt["added_entries"] == 1


@pytest.fixture
def agent(episode, monkeypatch):
    args = parser().parse_args(["--supervision-mode", "astra-only", "--max-calls", "8", "--tool-timeout", "0.01"])
    model = runner.Agent(episode.directory, object(), args)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(bridge, "verify", lambda _: None)
    yield model
    if not model.lock.closed:
        model.lock.close()


def test_api_retry_is_bounded_and_counts_attempts(agent, monkeypatch):
    attempts = []

    def fail(*_args, **_kwargs):
        attempts.append(1)
        raise ConnectionError("MUST_NOT_RECORD_CREDENTIALS")

    monkeypatch.setattr(runner, "request", fail)
    monkeypatch.setattr(runner, "retryable", lambda _: True)
    assert agent.run() == 1
    assert len(attempts) == agent.state["calls"] == 3
    assert agent.tools.status()["result"]["lifecycle"] == "closed"
    assert "MUST_NOT_RECORD_CREDENTIALS" not in (agent.folder / "api-errors.jsonl").read_text()


def test_nonretryable_api_error_closes_episode(agent, monkeypatch):
    def fail(*_args, **_kwargs):
        raise ValueError("invalid authentication")

    monkeypatch.setattr(runner, "request", fail)
    monkeypatch.setattr(runner, "retryable", lambda _: False)
    assert agent.run() == 1
    assert agent.state["calls"] == 1
    assert agent.tools.status()["result"]["lifecycle"] == "closed"


@pytest.fixture
def api_status_error(monkeypatch):
    # The deterministic harness suite also runs without the optional API SDK installed.
    class APIStatusError(Exception):
        def __init__(self, status, body):
            super().__init__("private server body")
            self.status_code = status
            self.body = body

    monkeypatch.setitem(
        sys.modules, "openai", SimpleNamespace(APIConnectionError=ConnectionError, APIStatusError=APIStatusError)
    )
    return APIStatusError


@pytest.mark.parametrize("status", [400, 408, 409, 429, 500, 503])
def test_http_errors_retry_with_backoff_without_controls(agent, monkeypatch, episode, status, api_status_error):
    attempts = []
    sleeps = []
    sentinel = SimpleNamespace(model_dump=lambda **_: {})

    def flaky(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise api_status_error(status, {"message": "MUST_NOT_RECORD_CREDENTIALS"})
        return sentinel

    monkeypatch.setattr(runner, "request", flaky)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    assert agent.call_api() is sentinel
    assert sleeps == [5, 15]
    assert agent.state["calls"] == 3
    assert not episode.runtime.calls
    assert "MUST_NOT_RECORD_CREDENTIALS" not in (agent.folder / "api-errors.jsonl").read_text()


@pytest.mark.parametrize(("status", "body"), [(402, {}), (429, {"code": "insufficient_quota"})])
def test_exhausted_quota_does_not_retry(agent, monkeypatch, api_status_error, status, body):
    def fail(*_args, **_kwargs):
        raise api_status_error(status, body)

    sleeps = []
    monkeypatch.setattr(runner, "request", fail)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    with pytest.raises(api_status_error):
        agent.call_api()
    assert agent.state["calls"] == 1
    assert not sleeps
    error = json.loads((agent.folder / "api-errors.jsonl").read_text())
    assert error["category"] == "quota_exhausted"
    assert error["retry"] is False


def test_encrypted_reasoning_error_retries_original_evidence_and_tool_ids(
    agent, monkeypatch, episode, api_status_error
):
    evidence = [
        {"role": "user", "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAA"}]},
        {"type": "function_call", "call_id": "executed", "name": "release", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "executed", "output": "already executed"},
    ]
    original = [{"type": "reasoning", "encrypted_content": "invalid"}, *evidence]
    agent.state["messages"] = original
    captured = []
    sentinel = SimpleNamespace(model_dump=lambda **_: {})

    def request(*_args, **kwargs):
        captured.append(kwargs["messages"])
        if len(captured) < 3:
            raise api_status_error(400, {"code": "invalid_encrypted_content"})
        return sentinel

    monkeypatch.setattr(runner, "request", request)
    assert agent.call_api() is sentinel
    assert captured == [original, original, original]
    assert agent.state["messages"] == original
    assert not list(agent.folder.glob("reasoning-before-recovery-*.json"))
    assert not episode.runtime.calls
    assert not agent.tools.state["image_deliveries"]


def test_repeated_http_400_exits_after_three_attempts(agent, monkeypatch, api_status_error):
    def fail(*_args, **_kwargs):
        raise api_status_error(400, {"code": "invalid_request"})

    monkeypatch.setattr(runner, "request", fail)
    assert agent.run() == 1
    assert agent.state["calls"] == 3
    assert agent.tools.status()["result"]["lifecycle"] == "closed"


def test_partial_api_response_never_executes_tools(agent, monkeypatch, episode):
    payload = {
        "status": "incomplete",
        "output": [{"type": "function_call", "name": "intervene", "call_id": "partial", "arguments": "{"}],
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    monkeypatch.setattr(runner, "request", lambda *_args, **_kwargs: SimpleNamespace(model_dump=lambda **_: payload))
    assert agent.run() == 1
    assert agent.state["stop_reason"] == "incomplete_api_response"
    assert not episode.runtime.calls


@pytest.mark.parametrize(("limit", "incoming"), [(300_000, 120_000), (500_000, 400_000)])
def test_context_budget_does_not_limit_cumulative_token_usage(agent, monkeypatch, limit, incoming):
    assert agent.args.max_context_tokens == 300_000
    agent.args.max_context_tokens = limit
    agent.state["input_tokens"] = 10_000_000
    agent.state["context_tokens"] = incoming
    payload = {"status": "incomplete", "output": [], "usage": {"input_tokens": incoming, "output_tokens": 0}}
    monkeypatch.setattr(runner, "request", lambda *_args, **_kwargs: SimpleNamespace(model_dump=lambda **_: payload))
    assert agent.run() == 1
    assert agent.state["calls"] == 1
    assert agent.state["input_tokens"] == 10_000_000 + incoming
    assert agent.state["stop_reason"] == "incomplete_api_response"


@pytest.mark.parametrize("pending", [False, True])
def test_over_context_response_never_executes_tools_even_on_resume(agent, monkeypatch, episode, pending):
    payload = {
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "name": "intervene",
                "call_id": "over-context",
                "arguments": json.dumps(intervention(agent.tools)),
            }
        ],
        "usage": {"input_tokens": 299_999, "output_tokens": 2},
    }
    if pending:
        agent.state["pending_response"] = payload
    monkeypatch.setattr(runner, "request", lambda *_args, **_kwargs: SimpleNamespace(model_dump=lambda **_: payload))
    processed = []
    monkeypatch.setattr(agent, "process_response", lambda data: processed.append(data))
    assert agent.run() == 1
    assert agent.state["stop_reason"] == "api_context_budget"
    assert agent.state["context_tokens"] == 300_001
    assert not processed
    assert not episode.runtime.calls
    assert agent.tools.status()["result"]["lifecycle"] == "closed"


def test_context_checkpoint_requests_memory_before_limit(agent, monkeypatch):
    agent.state["context_tokens"] = 240_000
    payload = {"status": "incomplete", "output": [], "usage": {"input_tokens": 240_100, "output_tokens": 0}}
    messages = []

    def request(*_args, **kwargs):
        messages.extend(kwargs["messages"])
        return SimpleNamespace(model_dump=lambda **_: payload)

    monkeypatch.setattr(runner, "request", request)
    assert agent.run() == 1
    assert "Context checkpoint due" in json.dumps(messages)
    assert agent.state["stop_reason"] == "incomplete_api_response"


@pytest.mark.parametrize(("context_tokens", "compacted"), [(120_000, False), (239_999, False), (240_000, True)])
def test_only_context_threshold_can_replace_prefix_after_memory(agent, monkeypatch, context_tokens, compacted):
    tools = agent.tools
    tools.execute("history", {"first": 0, "last": 0}, "history")
    tools.acknowledge_images()
    tools.execute(
        "write_memory",
        {"text": "Frame 0: counts=0; identity uncertain.", "frames": [0], "candidate_id": None},
        "memory",
    )
    original = tools.image(tools.status()["latest"]["rgb"]["front"])
    agent.state["messages"].append(
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "PREFIX_SENTINEL"},
                *[runner.image_content(original) for _ in range(45)],
            ],
        }
    )
    agent.state.update(
        context_tokens=context_tokens,
        since_compaction=999,
        last_compaction_frame=-1000,
        last_successful_tools=["write_memory"],
    )
    captured = []

    def request(*_args, **kwargs):
        captured.append(json.dumps(kwargs["messages"]))
        payload = {"status": "incomplete", "output": [], "usage": {"input_tokens": 1000, "output_tokens": 0}}
        return SimpleNamespace(model_dump=lambda **_: payload)

    monkeypatch.setattr(runner, "request", request)
    assert agent.run() == 1
    assert ("PREFIX_SENTINEL" not in captured[0]) == compacted
    assert bool(list(agent.folder.glob("context-before-*.json"))) == compacted


def test_legacy_resume_recovers_context_usage_without_using_cumulative_tokens(agent, monkeypatch):
    agent.resumed = True
    del agent.state["context_tokens"]
    agent.state.update(calls=1, since_compaction=1)
    rgb.write_json(agent.folder / "response-0001.json", {"usage": {"input_tokens": 300_001, "output_tokens": 0}})
    monkeypatch.setattr(
        runner, "request", lambda *_args, **_kwargs: pytest.fail("over-limit context must not call API")
    )
    assert agent.run() == 1
    assert agent.state["calls"] == 1
    assert agent.state["stop_reason"] == "api_context_budget"


@pytest.mark.parametrize("updates_enabled", [False, True])
def test_compaction_keeps_memory_state_and_blocks_undelivered_images(agent, updates_enabled):
    tools = agent.tools
    tools.manifest["experience_update"] = updates_enabled
    tools.execute("history", {"first": 0, "last": 0}, "history")
    with pytest.raises(ValueError, match="undelivered"):
        agent.compact()
    tools.acknowledge_images()
    tools.execute(
        "write_memory",
        {"text": "Frame 0: no button press yet; count=0; identity uncertain.", "frames": [0], "candidate_id": None},
        "memory",
    )
    agent.state["messages"].append(runner.user_message("OLD_CONTEXT"))
    agent.state.update(context_tokens=250_000, input_tokens=4_000_000)
    agent.compact()
    assert agent.state["context_tokens"] is None
    assert agent.state["input_tokens"] == 4_000_000
    restored = json.loads(agent.state["messages"][0]["content"][0]["text"])
    assert restored["experience_update"] is updates_enabled
    context = json.dumps(agent.state["messages"])
    assert "count=0" in context
    assert "Official task" in context
    assert "expected_frame" in context
    assert "OLD_CONTEXT" not in context
    assert "OLD_CONTEXT" in (agent.folder / "context-before-0000.json").read_text()


def test_one_tool_per_response_prevents_unseen_image_actions(agent, episode):
    calls = [
        {"type": "function_call", "name": "history", "call_id": "a", "arguments": '{"first":0,"last":0}'},
        {"type": "function_call", "name": "finish", "call_id": "b", "arguments": '{"reason":"test"}'},
    ]
    agent.process_response({"output": calls})
    assert not episode.supervisor.closed
    assert not agent.tools.state["image_deliveries"]


def test_resume_lock_and_tool_result_replay(agent, episode):
    with pytest.raises(BlockingIOError):
        runner.Agent(episode.directory, object(), agent.args)
    call = {"type": "function_call", "name": "status", "call_id": "stable", "arguments": "{}"}
    agent.state["pending_response"] = {"output": [call]}
    agent.state["messages"].append(call)
    agent.save()
    agent.lock.close()
    resumed = runner.Agent(episode.directory, object(), agent.args)
    try:
        assert resumed.resumed
        resumed.process_response(resumed.state["pending_response"])
        resumed.process_response({"output": [call]})
        outputs = [m for m in resumed.state["messages"] if m.get("type") == "function_call_output"]
        assert len(outputs) == 1
        assert not episode.runtime.calls
    finally:
        resumed.lock.close()


def test_full_agent_smoke_closes_reviews_and_verifies(agent, monkeypatch, episode):
    agent.args.max_calls = 30
    agent.args.smoke = True
    stages = iter(
        [
            "history",
            "view_image",
            "set_subgoal",
            "release",
            "history",
            "intervene",
            "history",
            "write_memory",
            "write_review",
            "complete",
        ]
    )
    verified = []

    def fake_api(*_args, **kwargs):
        # Production transport really includes image bytes in the next request.
        if agent.state["calls"] == 2:
            output = kwargs["messages"][-1]["output"]
            assert output[1]["type"] == "input_image"
            assert output[1]["image_url"].startswith("data:image/jpeg;base64,")
        name = next(stages)
        current = identity(agent.tools)
        frame = current["expected_frame"]
        if name == "history":
            args = {"first": 0, "last": frame}
        elif name == "view_image":
            args = {"frame": frame, "camera": "front"}
        elif name == "set_subgoal":
            args = {**current, "subgoal": SPEC, "reason": "visible target"}
        elif name == "release":
            args = {**current, "prefix": 1, "reason": "reviewed candidate"}
        elif name == "intervene":
            args = intervention(agent.tools)
        elif name in ("write_memory", "write_review"):
            args = {"text": "Observed changed pose and new candidate; task not complete.", "frames": [frame]}
            if name == "write_memory":
                args["candidate_id"] = current["candidate_id"]
            else:
                assert episode.supervisor.closed
        else:
            args = {}
        data = {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "name": name,
                    "call_id": f"call-{agent.state['calls']}",
                    "arguments": json.dumps(args),
                }
            ],
            "usage": {"input_tokens": 30, "output_tokens": 10},
        }
        return SimpleNamespace(model_dump=lambda **_: data)

    def verify(_):
        assert episode.supervisor.closed
        verified.append(True)

    monkeypatch.setattr(runner, "request", fake_api)
    monkeypatch.setattr(bridge, "verify", verify)
    assert agent.run() == 0
    assert len(episode.runtime.calls) == 2
    report = rgb.read_json(agent.folder / "report.json")
    assert report["takeover_smoke_passed"]
    assert report["review_written_by_model"]
    assert report["result"]["outcome"] == "incomplete"
    assert verified == [True]


def test_api_budget_and_interrupt_finalize_without_task_claims(agent, monkeypatch, episode):
    agent.state["calls"] = agent.args.max_calls
    assert agent.run() == 1
    assert episode.supervisor.closed
    assert not episode.runtime.calls
    assert "machine report" in (episode.directory / "review.md").read_text()


def test_api_retry_success_does_not_duplicate_controls(agent, monkeypatch, episode):
    attempts = []
    sentinel = SimpleNamespace(model_dump=lambda **_: {})

    def flaky(*_args, **_kwargs):
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("temporary")
        return sentinel

    monkeypatch.setattr(runner, "request", flaky)
    monkeypatch.setattr(runner, "retryable", lambda _: True)
    assert agent.call_api() is sentinel
    assert agent.state["calls"] == 3
    assert not episode.runtime.calls


def test_interrupt_finishes_original_episode(agent, monkeypatch, episode):
    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "request", interrupt)
    assert agent.run() == 1
    assert episode.supervisor.closed
    assert not episode.runtime.calls
    assert rgb.read_json(agent.folder / "report.json")["error_type"] == "KeyboardInterrupt"


def test_jev_advice_keeps_astra_delivery_separate(episode, monkeypatch):
    from scripts.robomme_agent.jev import VisualAdviser

    profile = Path(__file__).parent / "robomme_agent/profiles/jev27-local.json"
    tools = EpisodeTools(episode.directory, jev_config=profile)
    before = list(tools.state["seen"])
    calls = []

    def observe(self, question, images, accounting, **kwargs):
        calls.append(question)
        return {"status": "answered", "label": "uncertain", "review_required": True}

    monkeypatch.setattr(VisualAdviser, "observe", observe)
    args = {"expected_frame": tools.status()["latest"]["frame_id"], "question": "contact_readiness", "camera": "wrist"}
    result, images = tools.execute("jev_observe", args, "jev-test")
    assert result["review_required"]
    assert not images
    assert tools.state["seen"] == before
    assert not tools.state["image_deliveries"]
    tools.execute("jev_observe", args, "jev-test")
    assert calls == ["contact_readiness"]
    with pytest.raises(ValueError, match="stale"):
        tools.execute("jev_observe", {**args, "expected_frame": args["expected_frame"] + 1}, "stale")
    with pytest.raises(ValueError, match="original JEV"):
        EpisodeTools(episode.directory)


@pytest.mark.parametrize("already_closed", [False, True])
def test_large_reset_history_is_batched_without_early_compaction_or_forging_review(
    agent, episode, monkeypatch, already_closed
):
    agent.args.max_calls = 12
    for _ in range(300):
        episode.recorder.capture({"robot": episode.runtime.robot}, {}, "reset", rgb.safe_feedback({}))
    episode.supervisor.publish()
    if already_closed:
        agent.tools.finish("test terminal history delivery")
    counts = []

    def fake_api(*_, **kwargs):
        counts.append(runner.image_count(kwargs["messages"]))
        delivered = [f for item in agent.tools.state["image_deliveries"].values() for f in item["frames"]]
        if agent.tools.status()["unseen_count"] or agent.tools.status()["result"]["lifecycle"] != "closed":
            if delivered and max(delivered) == 300:
                if already_closed:
                    name, args = "write_review", {"text": "Terminal history fully viewed.", "frames": [300]}
                else:
                    name, args = "finish", {"reason": "all reset history inspected"}
            else:
                assert delivered
                # A future physical decision is still forbidden while reset history remains unseen.
                name, args = (
                    "write_memory",
                    {
                        "text": f"Reviewed reset through {max(delivered)}; remaining history pending.",
                        "frames": [max(delivered)],
                        "candidate_id": None,
                    },
                )
        elif not agent.tools.state["review_written"]:
            name, args = "write_review", {"text": "Complete reset viewed; no controls executed.", "frames": [300]}
        else:
            name, args = "complete", {}
        payload = {
            "status": "completed",
            "usage": {},
            "output": [
                {
                    "type": "function_call",
                    "name": name,
                    "arguments": json.dumps(args),
                    "call_id": f"large-reset-{agent.state['calls']}",
                }
            ],
        }
        return SimpleNamespace(model_dump=lambda **_: payload)

    monkeypatch.setattr(runner, "request", fake_api)
    assert agent.run() == 0
    assert set(agent.tools.state["seen"]) == set(range(301))
    assert not episode.runtime.calls
    assert len(counts) == (4 if already_closed else 5)
    assert max(counts) > 40
    assert not list(agent.folder.glob("context-before-*.json"))


def test_image_limit_recovery_preserves_new_images_and_requests_checkpoint(agent, monkeypatch):
    agent.tools.supervision_mode = "step-budget"
    path = agent.tools.image(agent.tools.status()["latest"]["rgb"]["front"])
    old = {"role": "user", "content": [runner.image_content(path)]}
    agent.state["messages"].append(old)
    agent.state["acknowledged_message_count"] = len(agent.state["messages"])
    agent.state["messages"].append({"role": "user", "content": [runner.image_content(path)]})
    agent.save()
    attempts = []

    class LimitError(Exception):
        body = "maximum number of images reached"
        status_code = 400

    def request(*_args, **kwargs):
        attempts.append(runner.image_count(kwargs["messages"]))
        if len(attempts) == 1:
            raise LimitError("secret must never enter archives")
        return SimpleNamespace(model_dump=lambda **_: {"status": "incomplete", "output": [], "usage": {}})

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner, "retryable", lambda _: False)
    assert agent.call_api() is not None
    assert attempts == [2, 1]
    assert agent.state["image_limit_checkpoint"]
    assert agent.tools.state["memory_checkpoint_required"]
    assert agent.tools.state["seen"] == []
    archive = next(agent.folder.glob("images-before-limit-*.json"))
    assert runner.image_count(rgb.read_json(archive)) == 2
    assert "secret" not in (agent.folder / "api-errors.jsonl").read_text()


def test_image_limit_recovery_does_not_discard_never_acknowledged_images(agent):
    path = agent.tools.image(agent.tools.status()["latest"]["rgb"]["front"])
    agent.state["messages"].append({"role": "user", "content": [runner.image_content(path)]})
    agent.state["acknowledged_message_count"] = 0
    assert not agent.recover_image_limit()
    assert runner.image_count(agent.state["messages"]) == 1


def test_closed_review_recovery_never_exposes_physical_tools(agent, episode):
    from scripts.run_robomme_agent import review_directory

    agent.tools.finish("closed for review")
    agent.save()
    agent.lock.close()
    agent.args.resume_review = episode.directory.parent
    # Fixture artifacts live directly in episode.directory; test CLI helper on a separate run layout.
    recovered = runner.Agent(episode.directory, object(), agent.args)
    try:
        names = {t["name"] for t in recovered.available_tools}
        assert {"write_review", "experience_commit", "history", "complete"} <= names
        assert not {"release", "intervene", "set_subgoal", "schedule_vla", "finish", "query"} & names
        assert recovered.tools.status()["result"]["lifecycle"] == "closed"
        assert not episode.runtime.calls
        with pytest.raises(FileNotFoundError):
            review_directory(episode.directory)
    finally:
        recovered.lock.close()


def test_image_rejection_checkpoint_compacts_only_after_model_memory(agent, monkeypatch):
    see_current(agent.tools)
    agent.state["context_tokens"] = 1000
    agent.state["image_limit_checkpoint"] = True
    agent.tools.state["memory_checkpoint_required"] = True
    captured = []

    def request(*_args, **kwargs):
        captured.append(json.dumps(kwargs["messages"]))
        payload = {
            "status": "completed" if len(captured) == 1 else "incomplete",
            "output": [
                {
                    "type": "function_call",
                    "name": "write_memory",
                    "call_id": "limit-memory",
                    "arguments": json.dumps(
                        {
                            "text": "Checkpoint: task unresolved; preserve uncertainty.",
                            "frames": [0],
                            "candidate_id": None,
                        }
                    ),
                }
            ]
            if len(captured) == 1
            else [],
            "usage": {"input_tokens": 1000, "output_tokens": 100},
        }
        return SimpleNamespace(model_dump=lambda **_: payload)

    monkeypatch.setattr(runner, "request", request)
    assert agent.run() == 1  # Deliberately incomplete mock response after the checkpoint.
    assert "Context checkpoint due" in captured[0]
    assert "Context compacted" in captured[1]
    assert "Checkpoint: task unresolved" in captured[1]
    assert not agent.state.get("image_limit_checkpoint")
    assert not agent.tools.state["memory_checkpoint_required"]
    assert list(agent.folder.glob("context-before-*.json"))
