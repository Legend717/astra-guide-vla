"""Crash boundaries and outcome-gated learning, using only local deterministic fixtures."""

import json
from types import SimpleNamespace

import pytest

from scripts import codex_rgb_eval as rgb
from scripts import run_robomme_agent as entrypoint
from scripts.codex_supervisor import experience
from scripts.robomme_agent import runner
from scripts.robomme_agent.scheduling_test import args as schedule_args
from scripts.robomme_agent.scheduling_test import scheduled  # noqa: F401
from scripts.robomme_agent.supervision_test import contract
from scripts.robomme_agent.supervision_test import delegated  # noqa: F401
from scripts.robomme_agent.tools import EpisodeTools
from scripts.robomme_agent_test import agent  # noqa: F401
from scripts.robomme_agent_test import episode  # noqa: F401
from scripts.run_robomme_agent import parser


class SimulatedCrash(BaseException):
    """Bypass ordinary tool errors, as an abrupt process exit would."""


@pytest.mark.parametrize("boundary", ["tool_saved", "output_saved"])
@pytest.mark.parametrize("legacy", [False, True])
def test_response_replay_never_acknowledges_new_tool_images(agent, episode, monkeypatch, boundary, legacy):  # noqa: F811
    call = {"type": "function_call", "name": "history", "call_id": "crash-history", "arguments": '{"first":0,"last":0}'}
    payload = {"status": "completed", "output": [call], "usage": {"input_tokens": 10, "output_tokens": 1}}
    agent.state["messages"].append(call)
    agent.state["pending_response"] = payload
    if not legacy:
        agent.state["request_image_delivery_ids"] = []
    agent.save()
    with monkeypatch.context() as patch:
        if boundary == "tool_saved":
            original = agent.tools.execute

            def execute(*args):
                original(*args)
                raise SimulatedCrash

            patch.setattr(agent.tools, "execute", execute)
        else:
            original_save = agent.save

            def save():
                original_save()
                if any(m.get("type") == "function_call_output" for m in agent.state["messages"]):
                    raise SimulatedCrash

            patch.setattr(agent, "save", save)
        with pytest.raises(SimulatedCrash):
            agent.process_response(payload)
    assert agent.tools.state["image_deliveries"]
    assert not agent.tools.state["seen"]
    agent.lock.close()
    recovered = runner.Agent(episode.directory, object(), agent.args)
    calls = []

    def request(*_args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            # These images have been offered/replayed, but this request has not returned yet.
            assert not recovered.tools.state["seen"]
            assert runner.request_image_deliveries(kwargs["messages"])
            data = {
                "status": "completed",
                "output": [{"type": "function_call", "name": "status", "call_id": "read", "arguments": "{}"}],
                "usage": {"input_tokens": 20, "output_tokens": 1},
            }
        else:
            assert recovered.tools.state["seen"] == [0]
            data = {"status": "incomplete", "output": [], "usage": {"input_tokens": 20, "output_tokens": 1}}
        return SimpleNamespace(model_dump=lambda **_: data)

    monkeypatch.setattr(runner, "request", request)
    assert recovered.run() == 1
    assert len(calls) == 2
    assert recovered.tools.state["seen"] == [0]
    assert not episode.runtime.calls


def test_only_image_deliveries_attached_to_request_are_acknowledged(agent, monkeypatch):  # noqa: F811
    agent.tools.handoff("orphan")  # Crash before adding the offer to the conversation.
    bundle, images = agent.tools.handoff("attached")
    agent.state["messages"].append(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": json.dumps(bundle)}, *[runner.image_content(p) for p in images]],
        }
    )
    response = SimpleNamespace(model_dump=lambda **_: {"usage": {}})
    monkeypatch.setattr(runner, "request", lambda *_args, **_kwargs: response)
    agent.call_api()
    assert agent.state["request_image_delivery_ids"] == ["attached"]
    assert not agent.tools.state["seen"]
    agent.tools.acknowledge_images(agent.state["request_image_delivery_ids"])
    assert not agent.tools.state["image_deliveries"]
    agent.tools.acknowledge_images(agent.state["request_image_delivery_ids"])
    assert not agent.tools.state["image_deliveries"]


@pytest.mark.parametrize("fixture", ["scheduled", "delegated"])
@pytest.mark.parametrize("pending_saved", [False, True])
def test_cancel_prepared_child_never_enqueues_it(request, monkeypatch, fixture, pending_saved):
    tools, harness = request.getfixturevalue(fixture)
    name = "schedule_vla" if fixture == "scheduled" else "supervise_vla"
    args = schedule_args(tools) if fixture == "scheduled" else contract(tools)

    def stop_before_enqueue(_operation, _args, identifier):
        if pending_saved:
            tools.state["pending"] = identifier
            tools.save()
        raise SimulatedCrash

    with monkeypatch.context() as patch:
        patch.setattr(tools, "submit", stop_before_enqueue)
        with pytest.raises(SimulatedCrash):
            tools.execute(name, args, "prepared")
    recovered = EpisodeTools(harness.directory, timeout=0.001, supervision_mode=tools.supervision_mode)
    recovered.finish("cancel prepared work")
    assert not harness.runtime.calls
    assert not recovered.state["pending"]
    saved = next(iter(recovered.state["delegations"].values()))
    assert saved["wake_reasons"] == ["delegation_cancelled"]
    assert saved["next_request"] is None
    assert saved["steps"] == 0
    assert not list((harness.directory / "requests").glob(f"{saved['id']}-c*.json"))


@pytest.mark.parametrize("fixture", ["scheduled", "delegated"])
@pytest.mark.parametrize("receipt_ready", [False, True])
def test_cancel_enqueued_child_settles_once(request, fixture, receipt_ready):
    tools, harness = request.getfixturevalue(fixture)
    harness.automatic = False
    name = "schedule_vla" if fixture == "scheduled" else "supervise_vla"
    args = schedule_args(tools) if fixture == "scheduled" else contract(tools)
    result, _ = tools.execute(name, args, "enqueued")
    if receipt_ready:
        harness.process(result["request_id"])
    recovered = EpisodeTools(harness.directory, timeout=0.001, supervision_mode=tools.supervision_mode)
    harness.automatic = True
    recovered.finish("cancel enqueued work")
    expected = 16 if fixture == "scheduled" else 2
    assert len(harness.runtime.calls) == expected
    saved = next(iter(recovered.state["delegations"].values()))
    assert saved["steps"] == expected
    assert saved["wake_reasons"] == ["delegation_cancelled"]
    assert len(list((harness.directory / "requests").glob(f"{saved['id']}-c*.json"))) == 1


@pytest.mark.parametrize(("mode", "prefix"), [("ordinary", 2), ("timing", 4)])
@pytest.mark.parametrize("inside", [False, True])
def test_legacy_gripper_guard_only_checks_executable_prefix(delegated, mode, prefix, inside):  # noqa: F811
    tools, harness = delegated
    harness.supervisor.prefix = prefix
    candidate = harness.supervisor.candidate
    candidate["default_prefix"] = prefix
    trajectory = candidate["eef"]["trajectory"]
    trajectory[1 if inside else prefix]["gripper_command"] = -1
    harness.supervisor.publish()
    result, _ = tools.execute("supervise_vla", contract(tools, phase_mode=mode, max_chunks=1), "prefix")
    assert result["executed_steps"] == (0 if inside else prefix)
    assert ("candidate_gripper_transition" in result["wake_reasons"]) == inside


@pytest.mark.parametrize("scope", ["global", "task"])
@pytest.mark.parametrize("outcome", ["success", "failure", "timeout", "error", "incomplete", "unknown"])
def test_failure_only_experience_commit_gate(episode, scope, outcome):  # noqa: F811
    manifest = rgb.read_json(episode.directory / "manifest.json")
    manifest.update(
        experience_update_policy="failure-only",
        task_experience_input={"enabled": False, "source": str(episode.directory / "task.json")},
    )
    rgb.write_json(episode.directory / "manifest.json", manifest)
    result = rgb.read_json(episode.directory / "result.json")
    result.update(lifecycle="closed", outcome=outcome, stop_reason="user_finish")
    rgb.write_json(episode.directory / "result.json", result)
    submission = {
        "scope": scope,
        "entries": [
            {"claim": "Observed stall", "conditions": "mock", "confidence": "low", "kind": "engineering", "frames": [0]}
        ],
    }
    target = episode.directory / ("task.json" if scope == "task" else "engineering.json")
    if outcome in ("success", "unknown"):
        with pytest.raises(ValueError, match="failure-only"):
            experience.commit(episode.directory, submission)
        assert not target.exists()
    else:
        first = experience.commit(episode.directory, submission)
        second = experience.commit(episode.directory, submission)
        assert first == second
        assert len(rgb.read_json(target)["entries"]) == 1
    tools = EpisodeTools(episode.directory)
    assert tools.status()["experience_updates"]["allowed"] == (outcome not in ("success", "unknown"))


def test_experience_policy_defaults_and_master_switch():
    args = parser().parse_args([])
    assert args.experience_update_policy == "always"
    assert (
        parser().parse_args(["--experience-update-policy", "failure-only"]).experience_update_policy == "failure-only"
    )
    result = {"lifecycle": "closed", "outcome": "success"}
    assert experience.update_policy({"experience_update": True}, result)["allowed"]
    for policy in ("always", "failure-only"):
        assert not experience.update_policy({"experience_update": False, "experience_update_policy": policy}, result)[
            "allowed"
        ]
    manifest = {"experience_update": True, "experience_update_policy": "failure-only"}
    assert not experience.update_policy(manifest, {"lifecycle": "ready", "outcome": "failure"})["allowed"]
    assert not experience.update_policy(manifest, {**result, "outcome": "error", "feedback": {"success": True}})[
        "allowed"
    ]


@pytest.mark.parametrize("policy", ["always", "failure-only"])
def test_launcher_forwards_experience_policy_without_starting_simulator(tmp_path, monkeypatch, policy):
    run = tmp_path / "new-run"
    args = parser().parse_args(["--run-dir", str(run), "--experience-update-policy", policy, "--max-steps", "100"])
    commands = []

    def fake_popen(command, **_kwargs):
        commands.append(command)
        directory = run / "artifacts"
        directory.mkdir(parents=True)
        rgb.write_json(directory / "result.json", {"lifecycle": "ready"})
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(entrypoint.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(entrypoint.tempfile, "tempdir", str(tmp_path))
    directory, process = entrypoint.launch(args)
    assert process.pid == 123
    command = commands[0]
    assert command[command.index("--experience-update-policy") + 1] == policy
    assert rgb.read_json(directory / "agent-host.json")["command"] == command


@pytest.mark.parametrize("success", [False, True])
def test_abandonment_and_success_use_official_outcome_at_tool_boundary(episode, success):  # noqa: F811
    manifest = rgb.read_json(episode.directory / "manifest.json")
    manifest["experience_update_policy"] = "failure-only"
    rgb.write_json(episode.directory / "manifest.json", manifest)
    tools = EpisodeTools(episode.directory, timeout=0.001)
    tools.handoff("initial")
    tools.acknowledge_images(["initial"])
    if success:
        episode.supervisor.state["outcome"] = "success"
        episode.supervisor.state["feedback"]["success"] = True
        episode.supervisor.publish()
    tools.finish("Astra gives up after reviewing the evidence")
    entry = {
        "claim": "Observed a limitation",
        "conditions": "mock",
        "confidence": "low",
        "kind": "engineering",
        "frames": [0],
    }
    if success:
        with pytest.raises(ValueError, match="official success"):
            tools.execute("experience_commit", {"entries": [entry]}, "commit")
        assert not (episode.directory / "engineering.json").exists()
    else:
        result, _ = tools.execute("experience_commit", {"entries": [entry]}, "commit")
        assert result["added_entries"] == 1
        assert tools.status()["result"]["outcome"] == "incomplete"


def test_resume_keeps_manifest_experience_policy(agent, episode):  # noqa: F811
    manifest = rgb.read_json(episode.directory / "manifest.json")
    manifest["experience_update_policy"] = "failure-only"
    rgb.write_json(episode.directory / "manifest.json", manifest)
    agent.lock.close()
    assert agent.args.experience_update_policy == "always"
    resumed = runner.Agent(episode.directory, object(), agent.args)
    try:
        assert resumed.tools.status()["experience_updates"]["policy"] == "failure-only"
        assert rgb.read_json(resumed.folder / "config.json")["experience_update_policy"] == "failure-only"
    finally:
        resumed.lock.close()
