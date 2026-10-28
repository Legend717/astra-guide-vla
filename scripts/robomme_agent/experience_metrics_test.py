"""Scoped evidence snapshots, immutable frozen identities and measured cost accounting."""

from types import SimpleNamespace

import pytest

from scripts import codex_rgb_eval as rgb
from scripts.codex_supervisor import experience
from scripts.robomme_agent import metrics
from scripts.robomme_agent_test import episode  # noqa: F401
from scripts.robomme_agent_test import tools_for


@pytest.mark.parametrize(
    ("calls", "backed_chunks", "vla_chunks", "expected_total"),
    [(79, 9, 49, 88), (79, 17, 58, 96), (7, 0, 3, 7)],
)
def test_astra_substitution_counts_executed_reviews_not_normal_answers(
    calls, backed_chunks, vla_chunks, expected_total
):
    operations = {
        "delegation_review_attribution_available": True,
        "jev_advice_backed_delegated_chunks": backed_chunks,
        "jev_normal_answers": 100,  # Multiple questions and unexecuted normal advice must not inflate savings.
        "vla_control_requests": vla_chunks,
    }
    result = metrics.astra_call_substitution(calls, operations)
    assert result["estimated_astra_calls_without_jev"] == expected_total
    assert result["estimated_astra_calls_replaced_by_jev"] == backed_chunks
    assert result["estimated_astra_call_replacement_fraction"] == pytest.approx(backed_chunks / expected_total)
    assert result["astra_call_fraction_with_jev"] == pytest.approx(calls / expected_total)
    assert result["astra_reviewed_vla_chunks"] == vla_chunks - backed_chunks
    assert result["jev_reviewed_vla_chunk_fraction"] == pytest.approx(backed_chunks / vla_chunks)


def test_astra_substitution_is_unknown_without_execution_attribution():
    result = metrics.astra_call_substitution(79, {"jev_normal_answers": 17})
    assert result["estimated_astra_calls_without_jev"] is None
    assert result["estimated_astra_calls_replaced_by_jev"] is None
    assert result["estimated_astra_call_replacement_fraction"] is None


def test_task_commits_are_isolated_versioned_and_freezable(episode):  # noqa: F811
    tools = tools_for(episode)
    task_path = episode.directory / "task.json"
    snap = experience.snapshot(episode.directory, task_path, enabled=True, filename="experience-task-input.json")
    tools.manifest["task_experience_input"] = snap
    rgb.write_json(episode.directory / "manifest.json", tools.manifest)
    result = rgb.read_json(episode.directory / "result.json")
    result["lifecycle"] = "closed"
    rgb.write_json(episode.directory / "result.json", result)
    entry = {
        "claim": "A measured candidate stalled",
        "conditions": "mock",
        "confidence": "low",
        "kind": "engineering",
        "frames": [0],
        "phase": "approach",
        "supervision": {"recommended_chunks": 2, "astra_watch": True, "evidence_count": 1},
    }
    first = experience.commit(episode.directory, {"scope": "task", "entries": [entry]})
    assert first["version"] == 1
    assert not (episode.directory / "engineering.json").exists()
    assert rgb.read_json(episode.directory / "experience-task-input.json")["entries"] == []
    identifier = rgb.read_json(task_path)["entries"][0]["id"]
    second = experience.commit(
        episode.directory,
        {
            "scope": "task",
            "entries": [
                {
                    **entry,
                    "claim": "New evidence contradicts the budget",
                    "supersedes": [identifier],
                    "status": "contradicted",
                }
            ],
        },
    )
    assert second["version"] == 2
    assert len(rgb.read_json(task_path)["entries"]) == 2
    tools.manifest["experience_update"] = False
    rgb.write_json(episode.directory / "manifest.json", tools.manifest)
    with pytest.raises(ValueError, match="disabled"):
        experience.commit(episode.directory, {"scope": "task", "entries": [entry]})


def test_migration_retains_source_and_stable_entry_ids(tmp_path):
    source = tmp_path / "old.json"
    original = {
        "version": 40,
        "entries": [
            {"claim": "InsertPeg insertion", "conditions": "peg", "source_run": "/absent"},
            {"claim": "StopCube counting", "conditions": "timing", "source_run": "/absent"},
            {"claim": "Check executed history", "conditions": "any task", "source_run": "/absent"},
        ],
    }
    rgb.write_json(source, original)
    before = source.read_bytes()
    result = experience.migrate(source, tmp_path / "new")
    assert source.read_bytes() == before
    assert [x["scope"] for x in result["mapping"]] == ["InsertPeg", "StopCube", "global"]
    assert rgb.read_json(tmp_path / "new/InsertPeg.json")["entries"][0]["id"] == "legacy-v40-0"
    with pytest.raises(ValueError, match="new"):
        experience.migrate(source, tmp_path / "new")


def test_metrics_count_retries_and_exclude_nested_execution_timing(episode):  # noqa: F811
    tools = tools_for(episode)
    for row in [
        {"event": "start", "call": 1, "attempt": 0, "phase": "execution"},
        {"event": "end", "call": 1, "ok": False, "seconds": 2},
        {"event": "start", "call": 2, "attempt": 1, "phase": "execution"},
        {"event": "end", "call": 2, "ok": True, "seconds": 3, "cached_input_tokens": 17},
        {"event": "start", "call": 3, "attempt": 0, "phase": "review"},
    ]:
        rgb.append_json(tools.agent_dir / "api-timing.jsonl", row)
    for row in [{"event": "control_return", "seconds": 4}, {"event": "execution_end", "seconds": 5}]:
        rgb.append_json(episode.directory / "events.jsonl", row)
    report = metrics.summarize(tools)
    assert report["api_retries"] == 1
    assert report["api_successes"] == 1
    assert report["api_unresolved_attempts"] == 1
    assert report["api_seconds"] == 5
    assert report["control_seconds"] == 4
    assert report["cached_input_tokens"] == 17


def test_execution_shares_use_real_partial_controls_without_double_counting(tmp_path):
    (tmp_path / "agent").mkdir()
    for kind in ("joint", "joint", "joint", "eef"):
        rgb.append_json(tmp_path / "events.jsonl", {"event": "control_return", "kind": kind})
    rgb.write_json(tmp_path / "agent/tools-state.json", {"delegations": {"one": {"steps": 2, "chunks": 1}}})
    for row in [
        {"status": "answered", "review_required": False},
        {"status": "answered", "review_required": True, "low_confidence": True},
        {"status": "answered", "review_required": True, "low_confidence": False},
        {"status": "unavailable"},
    ]:
        rgb.append_json(tmp_path / "agent/jev-observations.jsonl", row)
    result = metrics.execution_accounting(tmp_path, 4)
    assert result["control_ledger_matches_result"]
    assert result["gpt_control_fraction"] == 0.25
    assert result["vla_control_fraction"] == 0.75
    assert result["astra_direct_vla_fraction"] == 0.25
    assert result["delegated_step_fraction"] == 0.5
    assert result["delegated_chunks"] == 1
    assert result["jev_normal_answers"] == result["jev_alert_answers"] == result["jev_low_confidence_answers"] == 1
    assert result["jev_unavailable_answers"] == 1
    assert metrics.execution_accounting(tmp_path, 5)["gpt_control_fraction"] is None


def test_frozen_snapshot_rejects_drift(episode):  # noqa: F811
    tools = tools_for(episode)
    path = episode.directory / "frozen.json"
    tools.manifest.update(
        experience_update=False,
        experience_input={"sha256": "global"},
        task_experience_input={"sha256": "task"},
        policy={"checkpoint": "/checkpoint"},
    )
    settings = {
        "model": "test",
        "reasoning": "high",
        "max_delegated_chunks": 10,
        "prefix": 16,
        "max_steps": 1000,
        "max_calls": 80,
        "jev_config_sha256": None,
    }
    rgb.write_json(
        path,
        {
            "settings": settings,
            "experience_sha256": {"global": "global", "PickHighlight": "task"},
            "checkpoint": "/checkpoint",
        },
    )
    args = SimpleNamespace(**{k: v for k, v in settings.items() if k != "jev_config_sha256"}, frozen_manifest=path)
    assert metrics.validate_frozen(tools, args)["sha256"]
    args.max_delegated_chunks = 20
    with pytest.raises(ValueError, match="settings mismatch"):
        metrics.validate_frozen(tools, args)


def test_supervision_accounting_separates_budget_handoffs_and_zero_execution(tmp_path):
    (tmp_path / "agent").mkdir()
    rgb.write_json(
        tmp_path / "agent/tools-state.json",
        {
            "delegations": {
                "budget": {
                    "id": "budget",
                    "finished": True,
                    "steps": 32,
                    "chunks": 2,
                    "args": {"max_chunks": 6, "prefix_steps": 16, "checkpoint_steps": 32},
                    "wake_reasons": ["phase_checkpoint", "phase_checkpoint"],
                    "initial_review": "astra",
                },
                "alert": {"finished": True, "steps": 0, "chunks": 0, "wake_reasons": ["jev_alert"]},
                "pending": {"finished": False, "steps": 16, "chunks": 1},
            }
        },
    )
    for name, args in (("write_memory", {}), ("release", {"prefix": 4}), ("supervise_vla", {})):
        rgb.append_json(tmp_path / "agent/tools.jsonl", {"name": name, "args": args})
    result = metrics.supervision_accounting(tmp_path)
    assert result["astra_tool_dispatches"] == 3
    assert result["astra_release_prefix_counts"] == {"4": 1}
    assert result["delegation_contracts"] == 3
    assert result["delegation_finished_contracts"] == 2
    assert result["delegation_productive_contracts"] == result["delegation_zero_step_contracts"] == 1
    assert result["delegation_average_chunks_per_productive_contract"] == 2
    assert result["delegation_wake_counts"] == {"phase_checkpoint": 1, "jev_alert": 1}
    assert result["delegation_zero_step_wake_counts"] == {"jev_alert": 1}
    assert result["delegation_details"][0]["checkpoint_steps"] == 32
    assert result["delegation_details"][0]["initial_review"] == "astra"


def test_token_usage_separates_cumulative_usage_from_last_compacted_context(tmp_path):
    (tmp_path / "agent").mkdir()
    for call, incoming, cached, outgoing, reasoning in ((1, 100, 60, 20, 15), (3, 40, 30, 5, 2)):
        rgb.write_json(
            tmp_path / f"agent/response-{call:04d}.json",
            {
                "usage": {
                    "input_tokens": incoming,
                    "input_tokens_details": {"cached_tokens": cached},
                    "output_tokens": outgoing,
                    "output_tokens_details": {"reasoning_tokens": reasoning},
                }
            },
        )
    result = metrics.token_accounting(tmp_path)
    assert result["usage_response_count"] == 2
    assert result["total_tokens"] == 165
    assert result["uncached_input_tokens"] == 50
    assert result["reasoning_output_tokens"] == 17
    assert result["last_context_call"] == 3
    assert result["last_context_total_tokens"] == 45
    assert result["max_context_call"] == 1
    assert result["max_context_total_tokens"] == 120
    assert result["cache_write_tokens"] is None


def test_missing_final_usage_does_not_substitute_an_earlier_context(tmp_path):
    (tmp_path / "agent").mkdir()
    rgb.write_json(tmp_path / "agent/response-0001.json", {"usage": {"input_tokens": 10, "output_tokens": 2}})
    rgb.write_json(tmp_path / "agent/response-0002.json", {"usage": None})
    result = metrics.token_accounting(tmp_path)
    assert result["usage_missing_response_count"] == 1
    assert result["last_context_call"] == 2
    assert result["last_context_total_tokens"] is None
    assert result["uncached_input_tokens"] is None


def test_model_advice_attribution_excludes_astra_entry_and_unexecuted_partial_checks(tmp_path):
    (tmp_path / "agent").mkdir()
    for frame in range(4):
        rgb.append_json(
            tmp_path / "events.jsonl",
            {"event": "control_return", "kind": "joint", "request_id": f"d-c{frame // 2:03d}", "before_frame": frame},
        )
    rgb.write_json(
        tmp_path / "agent/tools-state.json",
        {"delegations": {"d": {"id": "d", "initial_review": "astra", "steps": 4, "chunks": 2}}},
    )
    for frame, review, low in ((2, False, False), (2, False, False), (4, False, False), (4, True, True)):
        rgb.append_json(
            tmp_path / "agent/jev-observations.jsonl",
            {
                "contract_id": "d",
                "frame_id": frame,
                "status": "answered",
                "review_required": review,
                "low_confidence": low,
            },
        )
    result = metrics.execution_accounting(tmp_path, 4)
    assert result["delegation_review_attribution_available"]
    assert result["astra_entry_delegated_steps"] == 2
    assert result["jev_advice_backed_delegated_steps"] == 2
    assert result["jev_advice_backed_delegated_chunks"] == 1
    assert result["jev_advice_backed_control_fraction"] == 0.5
    assert result["delegated_without_current_jev_review_steps"] == 2
