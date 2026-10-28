"""Step budgets and public-state guards using the real durable bridge, with no network/model."""

import pytest

from scripts import codex_rgb_eval as rgb
from scripts.codex_supervisor import experience
from scripts.robomme_agent import metrics
from scripts.robomme_agent.scheduling import orientation_distance
from scripts.robomme_agent.tools import EpisodeTools
from scripts.robomme_agent_test import episode  # noqa: F401
from scripts.robomme_agent_test import identity
from scripts.robomme_agent_test import initialize


@pytest.fixture
def scheduled(episode):  # noqa: F811
    episode.runtime.robot["gripper_state"] = [0.02, 0.02]
    episode.recorder.capture({"robot": episode.runtime.robot}, {}, "reset", rgb.safe_feedback({}))
    episode.supervisor.max_steps = 500
    episode.supervisor.prefix = 16
    episode.supervisor.fk.describe = lambda actions, current: {
        "trajectory": [{"target_xyz_rpy": current, "gripper_command": float(row[7])} for row in actions]
    }
    episode.supervisor.publish()
    tools = EpisodeTools(episode.directory, timeout=0.001, supervision_mode="step-budget")
    initialize(tools)
    return tools, episode


def args(tools, steps=100, **changes):
    return {**identity(tools), "budget_steps": steps, "reason": "test budget", **changes}


@pytest.mark.parametrize(("steps", "chunks", "last"), [(30, 2, 14), (100, 7, 4), (200, 13, 8)])
def test_exact_budgets_fresh_candidates_idempotency_and_sparse_handoff(scheduled, steps, chunks, last):
    tools, harness = scheduled
    request = args(tools, steps)
    before = tools.status()["latest"]["frame_id"]
    result, _ = tools.execute("schedule_vla", request, "schedule")
    assert result["executed_steps"] == len(harness.runtime.calls) == steps
    assert result["executed_chunks"] == chunks
    assert result["wake_reasons"] == ["step_budget"]
    contract = tools.state["delegations"][result["contract_id"]]
    assert contract["last_executed"]["args"]["prefix"] == last
    assert tools.status()["unseen_count"] == steps
    tools.execute("schedule_vla", request, "schedule")
    assert len(harness.runtime.calls) == steps
    with pytest.raises(ValueError, match="history"):
        tools.execute("release", {**identity(tools), "prefix": 1, "reason": "not yet seen"}, "unseen")
    bundle, images = tools.handoff("wake")
    assert bundle["delivered_history"] == [before + steps]
    assert len(images) == 2
    tools.acknowledge_images()
    assert tools.status()["review_required_count"] == 0
    assert tools.status()["unseen_count"] == steps - 1
    assert metrics.scheduling_accounting(tools.directory)["avoided_chunk_review_equivalents"] == chunks - 1


@pytest.mark.parametrize("steps", [0, 20, 31, 210, True, 30.0])
def test_invalid_budget_cannot_execute(scheduled, steps):
    tools, harness = scheduled
    with pytest.raises(ValueError, match="tool|budget"):
        tools.execute("schedule_vla", args(tools, steps), "invalid")
    assert not harness.runtime.calls


@pytest.mark.parametrize("pending", [False, True])
def test_per_chunk_review_stops_and_recovers_without_second_child(scheduled, pending):
    tools, harness = scheduled
    tools.review_every_chunk = True
    tools.state["supervision_config"]["review_every_chunk"] = True
    tools.save()
    request = args(tools, 100)
    harness.automatic = not pending
    result, _ = tools.execute("schedule_vla", request, "per-chunk")
    if pending:
        assert result["pending"]
        original = result["request_id"]
        harness.process(original)
        harness.automatic = True
        tools = EpisodeTools(harness.directory, timeout=0.001, supervision_mode="step-budget", review_every_chunk=True)
        result, _ = tools.execute("query", {"request_id": original}, "recover")
    assert result["executed_steps"] == len(harness.runtime.calls) == 16
    assert result["executed_chunks"] == 1
    assert result["remaining_steps"] == 84
    assert result["wake_reasons"] == ["per_chunk_review"]
    tools.execute("schedule_vla", request, "per-chunk")
    assert len(harness.runtime.calls) == 16
    assert not (harness.directory / "requests/per-chunk-c001.json").exists()
    with pytest.raises(ValueError, match="resume"):
        EpisodeTools(harness.directory, supervision_mode="step-budget")
    assert metrics.scheduling_accounting(tools.directory)["avoided_chunk_review_equivalents"] == 0


def test_pending_restart_preserves_child_and_budget(scheduled):
    tools, harness = scheduled
    harness.automatic = False
    result, _ = tools.execute("schedule_vla", args(tools, 30), "pending")
    assert result["pending"]
    original = result["request_id"]
    harness.process(original)
    harness.automatic = True
    recovered = EpisodeTools(harness.directory, timeout=0.001, supervision_mode="step-budget")
    result, _ = recovered.execute("query", {"request_id": original}, "recover")
    assert result["executed_steps"] == len(harness.runtime.calls) == 30
    recovered.execute("query", {"request_id": original}, "recover")
    assert len(harness.runtime.calls) == 30
    with pytest.raises(ValueError, match="resume"):
        EpisodeTools(harness.directory, supervision_mode="astra-only")


def test_tail_gripper_change_is_not_checked(scheduled):
    tools, harness = scheduled
    candidate = rgb.read_json(harness.directory / "decision.json")
    candidate["candidate"]["eef"]["trajectory"][-1]["gripper_command"] = -1
    rgb.write_json(harness.directory / "decision.json", candidate)
    result, _ = tools.execute(
        "schedule_vla", args(tools, 30, guards=[{"kind": "gripper_change", "when": "before_command"}]), "tail"
    )
    assert result["executed_steps"] == 30


def test_candidate_guard_stops_before_execution(scheduled):
    tools, harness = scheduled
    decision = rgb.read_json(harness.directory / "decision.json")
    decision["candidate"]["eef"]["trajectory"][2]["gripper_command"] = -1
    rgb.write_json(harness.directory / "decision.json", decision)
    result, _ = tools.execute(
        "schedule_vla", args(tools, guards=[{"kind": "gripper_change", "when": "before_command"}]), "guard"
    )
    assert result["wake_reasons"] == ["candidate_gripper_transition"]
    assert result["executed_steps"] == 0
    assert not harness.runtime.calls


def test_measured_gripper_returns_at_chunk_boundary_and_reanchors(scheduled):
    tools, harness = scheduled
    original = harness.runtime.step

    def step(action, kind):
        harness.runtime.robot["gripper_state"] = [0.01, 0.01]
        return original(action, kind)

    harness.runtime.step = step
    guards = [{"kind": "gripper_change"}]
    result, _ = tools.execute("schedule_vla", args(tools, guards=guards), "close")
    assert result["wake_reasons"] == ["measured_gripper_change"]
    assert result["executed_steps"] == 16
    tools.handoff("wake")
    tools.acknowledge_images()
    result, _ = tools.execute("schedule_vla", args(tools, 30, guards=guards), "renew")
    assert result["wake_reasons"] == ["step_budget"]
    assert result["executed_steps"] == 30


def test_stationary_is_opt_in_and_uses_controls(scheduled):
    tools, _ = scheduled
    result, _ = tools.execute("schedule_vla", args(tools, guards=[{"kind": "stationary"}]), "stationary")
    assert result["executed_steps"] == 32
    assert result["wake_reasons"] == ["robot_stationary"]
    tools.handoff("wake")
    tools.acknowledge_images()
    result, _ = tools.execute("schedule_vla", args(tools, 30), "wait")
    assert result["executed_steps"] == 30
    assert result["wake_reasons"] == ["step_budget"]


def test_direction_guard_and_multiple_reasons(scheduled):
    tools, harness = scheduled
    decision = rgb.read_json(harness.directory / "decision.json")
    decision["candidate"]["eef"]["trajectory"][3] = {"target_xyz_rpy": [-0.03, 0, 0.5, 0, 0, 0], "gripper_command": -1}
    rgb.write_json(harness.directory / "decision.json", decision)
    guards = [{"kind": "forbidden_direction", "direction": "-x"}, {"kind": "gripper_change", "when": "before_command"}]
    result, _ = tools.execute("schedule_vla", args(tools, guards=guards), "both")
    assert set(result["wake_reasons"]) == {"candidate_direction_limit", "candidate_gripper_transition"}
    assert len(result["guard_events"]) == 2
    assert not harness.runtime.calls


def test_partial_episode_end_counts_actual_controls(scheduled):
    tools, harness = scheduled
    harness.supervisor.max_steps = 19
    result, _ = tools.execute("schedule_vla", args(tools), "end")
    assert result["executed_steps"] == 19
    assert result["remaining_steps"] == 81
    assert "episode_ended" in result["wake_reasons"]


def test_sample_history_does_not_ack_omitted_frames(scheduled):
    tools, _ = scheduled
    tools.execute("schedule_vla", args(tools, 30), "move")
    result, images = tools.execute("history_sample", {"first": 2, "last": 31, "max_frames": 8}, "sample")
    assert len(result["selected_frames"]) == 8
    assert result["selected_frames"][0] == 2
    assert result["selected_frames"][-1] == 31
    assert images
    assert 2 not in tools.state["seen"]
    tools.acknowledge_images()
    assert set(range(2, 32)) - set(tools.state["seen"])
    result, _ = tools.execute("history_sample", {"first": 2, "last": 5}, "zoom")
    assert result["selected_frames"] == [2, 3, 4, 5]


@pytest.mark.parametrize(
    "advice", [{"max_trust_steps": 1}, {"close_review": True}, {"max_trust_steps": 110, "close_review": True}]
)
@pytest.mark.parametrize("operation", ["schedule_vla", "release"])
def test_loaded_experience_is_advisory_and_remains_in_context(scheduled, advice, operation):
    tools, harness = scheduled
    path = harness.directory / "source.json"
    rgb.write_json(
        path,
        {
            "entries": [
                {
                    "id": "cap",
                    "applicability": "step-budget",
                    "scheduling": {"subgoal": "press the button", **advice},
                }
            ]
        },
    )
    tools.manifest["task_experience_input"] = experience.snapshot(
        harness.directory, path, enabled=True, filename="experience-task-input.json"
    )
    context, _ = tools.execute("read_memory", {}, "context")
    assert context["task_experience"]["entries"][0]["scheduling"] == {"subgoal": "press the button", **advice}
    assert "subgoal_limit_steps" not in tools.status()["supervision"]
    before = len(harness.runtime.calls)
    request = args(tools) if operation == "schedule_vla" else {**identity(tools), "prefix": 16, "reason": "reviewed"}
    tools.execute(operation, request, "advisory")
    assert len(harness.runtime.calls) - before == (100 if operation == "schedule_vla" else 16)


def test_configured_execution_maximum_remains_enforced(scheduled):
    tools, harness = scheduled
    tools.max_scheduled_steps = 40
    with pytest.raises(ValueError, match="configured maximum"):
        tools.execute("schedule_vla", args(tools, 50), "too-large")
    assert not harness.runtime.calls
    result, _ = tools.execute("schedule_vla", args(tools, 40), "within")
    assert result["executed_steps"] == 40


def test_experience_is_automatically_injected_initially_and_after_compaction(scheduled):
    import json

    from scripts.robomme_agent import runner
    from scripts.run_robomme_agent import parser

    _, harness = scheduled
    manifest = rgb.read_json(harness.directory / "manifest.json")
    for key, filename in (
        ("experience_input", "experience-input.json"),
        ("task_experience_input", "experience-task-input.json"),
    ):
        source = harness.directory / ("source-" + filename)
        rgb.write_json(
            source,
            {
                "entries": [
                    {"id": key, "claim": "Review current evidence", "applicability": "step-budget"},
                    {"id": "jev-only", "claim": "JEV advice", "applicability": "jev"},
                ]
            },
        )
        manifest[key] = experience.snapshot(harness.directory, source, enabled=True, filename=filename)
    rgb.write_json(harness.directory / "manifest.json", manifest)
    model = runner.Agent(harness.directory, object(), parser().parse_args([]))
    initial = json.loads(model.state["messages"][0]["content"][0]["text"])
    for key, field in (("experience_input", "experience"), ("task_experience_input", "task_experience")):
        assert [e["id"] for e in initial["notes"][field]["entries"]] == [key]
    model.compact(allow_unseen=True)
    compacted = json.loads(model.state["messages"][0]["content"][0]["text"])
    for field in ("experience", "task_experience"):
        assert compacted["memory"][field] == initial["notes"][field]


def test_filtering_is_not_based_on_jev_provenance_and_supersession_is_scoped():
    entries = [
        {"id": "fact", "claim": "actual robot motion", "provenance": {"jev_model": "old"}},
        {"id": "old", "claim": "use JEV question", "applicability": "jev", "supersedes": ["fact"]},
        {"id": "budget", "applicability": "step-budget", "scheduling": {"subgoal": "press the button"}},
    ]
    view = experience.active_view({"entries": entries}, mode="step-budget")
    assert [e["id"] for e in view["entries"]] == ["fact", "budget"]
    assert [e["id"] for e in experience.active_view({"entries": entries}, mode="jev")["entries"]] == ["old"]


def test_orientation_wrap():
    assert orientation_distance([0, 0, 3.14159265], [0, 0, -3.14159265]) < 1e-6


def test_full_agent_loop_sparse_history_commit_and_costs(scheduled, monkeypatch):
    import json
    from types import SimpleNamespace

    from scripts.robomme_agent import runner
    from scripts.run_robomme_agent import parser

    tools, harness = scheduled
    source = harness.directory / "task-experience.json"
    manifest = rgb.read_json(harness.directory / "manifest.json")
    manifest["task_experience_input"] = experience.snapshot(
        harness.directory, source, enabled=True, filename="experience-task-input.json"
    )
    rgb.write_json(harness.directory / "manifest.json", manifest)
    options = parser().parse_args(["--max-calls", "12", "--tool-timeout", "0.01"])
    model = runner.Agent(harness.directory, object(), options)
    available = {tool["name"] for tool in model.available_tools}
    assert "schedule_vla" in available
    assert not {"jev_observe", "supervise_vla"} & available
    assert (
        "supervision"
        not in next(t for t in model.available_tools if t["name"] == "experience_commit")["parameters"]["properties"][
            "entries"
        ]["items"]["properties"]
    )
    index = 0

    def request(*_args, **_kwargs):
        nonlocal index
        index += 1
        latest = model.tools.status()["latest"]["frame_id"]
        operations = {
            1: ("schedule_vla", args(model.tools, 100)),
            2: ("history_sample", {"first": 2, "last": latest, "max_frames": 8}),
            3: ("finish", {"reason": "engineering test complete"}),
            4: ("write_review", {"text": "Mock control-budget and evidence delivery check", "frames": [latest]}),
            5: (
                "experience_commit",
                {
                    "scope": "task",
                    "entries": [
                        {
                            "claim": "Mock scheduling check",
                            "conditions": "mock only",
                            "confidence": "low",
                            "kind": "engineering",
                            "frames": [latest],
                            "scheduling": {"subgoal": "press the button", "max_trust_steps": 40},
                        }
                    ],
                },
            ),
            6: ("complete", {}),
        }
        name, arguments = operations[index]
        payload = {
            "status": "completed",
            "output": [
                {"type": "function_call", "name": name, "call_id": f"full-{index}", "arguments": json.dumps(arguments)}
            ],
            "usage": {"input_tokens": 100, "output_tokens": 10, "input_tokens_details": {"cached_tokens": 20}},
        }
        return SimpleNamespace(model_dump=lambda **_: payload)

    monkeypatch.setattr(runner, "request", request)
    monkeypatch.setattr(runner.bridge, "verify", lambda _: None)
    assert model.run() == 0
    assert model.state["calls"] == 6
    assert model.tools.status()["unseen_count"] > 0  # not a drain loop / not falsely acknowledged
    entries = rgb.read_json(source)["entries"]
    assert entries[0]["applicability"] == "step-budget"
    assert entries[0]["scheduling"]["max_trust_steps"] == 40
    assert rgb.read_json(harness.directory / "experience-task-input.json")["entries"] == []
    report = rgb.read_json(model.folder / "report.json")
    costs = report["metrics"]["scheduling_accounting"]
    assert costs["avoided_chunk_review_equivalents"] == 6
    assert costs["response_categories"] == {"decision": 1, "history_request": 1, "other": 1, "review": 3}
    assert costs["tokens_by_phase"]["review"]["context_total_tokens"] == 330


def test_duplicate_default_gripper_guard_is_rejected(scheduled):
    tools, _ = scheduled
    with pytest.raises(ValueError, match="duplicate"):
        tools.execute(
            "schedule_vla",
            args(tools, guards=[{"kind": "gripper_change"}, {"kind": "gripper_change", "when": "after_measured"}]),
            "duplicate",
        )


def test_stale_identity_and_missing_state_do_not_execute(scheduled):
    tools, harness = scheduled
    request = args(tools)
    request["expected_frame"] -= 1
    with pytest.raises(ValueError, match="stale"):
        tools.execute("schedule_vla", request, "stale")
    latest = rgb.read_json(harness.directory / "latest.json")
    latest["robot"]["gripper_state"] = []
    rgb.write_json(harness.directory / "latest.json", latest)
    result, _ = tools.execute("schedule_vla", args(tools), "missing-state")
    assert result["wake_reasons"] == ["invalid_robot_state"]
    assert not harness.runtime.calls


def test_single_original_is_citable_without_claiming_both_views_seen(scheduled):
    tools, harness = scheduled
    tools.execute("schedule_vla", args(tools, 30), "move")
    tools.execute("view_image", {"frame": 15, "camera": "wrist"}, "wrist")
    assert 15 not in tools.state["original_wrist"]
    tools.acknowledge_images()
    assert 15 in tools.state["original_wrist"]
    assert 15 not in tools.state["seen"]
    tools.execute(
        "write_memory",
        {"text": "Only wrist frame 15 viewed", "frames": [15], "candidate_id": identity(tools)["candidate_id"]},
        "cite",
    )
    tools.execute("view_image", {"frame": 15, "camera": "front"}, "front")
    tools.acknowledge_images()
    assert 15 in tools.state["seen"]
    assert tools.status()["review_required_count"] == 1  # old pair cannot authorize the current frame
    tools.execute("view_image", {"frame": 14, "camera": "wrist"}, "wrist-only")
    tools.acknowledge_images()
    assert 14 not in tools.state["seen"]
    tools.handoff("current")
    tools.acknowledge_images()
    tools.finish("test")
    entry = {
        "claim": "Wrist observation",
        "conditions": "mock",
        "confidence": "low",
        "kind": "engineering",
        "frames": [14],
    }
    result, _ = tools.execute("experience_commit", {"scope": "global", "entries": [entry]}, "commit")
    assert result["added_entries"] == 1
