"""Report measured latency and usage without double-counting nested execution timers."""

from collections import Counter
from datetime import datetime
import hashlib
import json
import time

from scripts.robomme_agent.jev import confidence_diagnostics


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def token_accounting(directory):
    """Use server usage, including image/reasoning tokens; final context is the last API input plus output."""
    responses = []
    for path in sorted((directory / "agent").glob("response-*.json")):
        response = json.loads(path.read_text())
        usage = response.get("usage") or {}
        incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
        valid = type(incoming) is int and type(outgoing) is int
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
        written = (usage.get("input_tokens_details") or {}).get("cache_write_tokens")
        reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
        responses.append(
            {
                "call": int(path.stem.removeprefix("response-")),
                "input_tokens": incoming,
                "cached_input_tokens": cached,
                "cache_write_tokens": written,
                "uncached_input_tokens": incoming - cached if valid and type(cached) is int else None,
                "output_tokens": outgoing,
                "reasoning_output_tokens": reasoning,
                "context_total_tokens": incoming + outgoing if valid else None,
            }
        )
    measured = [r for r in responses if r["context_total_tokens"] is not None]
    last = responses[-1] if responses else {}
    peak = max(measured, key=lambda r: r["context_total_tokens"]) if measured else {}

    def total(key):
        return sum(r[key] for r in measured) if measured and all(r[key] is not None for r in measured) else None

    return {
        "usage_response_count": len(measured),
        "usage_missing_response_count": len(responses) - len(measured),
        "total_tokens": total("context_total_tokens"),
        "uncached_input_tokens": total("uncached_input_tokens"),
        "cache_write_tokens": total("cache_write_tokens"),
        "reasoning_output_tokens": total("reasoning_output_tokens"),
        "last_context_call": last.get("call"),
        "last_context_input_tokens": last.get("input_tokens"),
        "last_context_cached_input_tokens": last.get("cached_input_tokens"),
        "last_context_output_tokens": last.get("output_tokens"),
        "last_context_total_tokens": last.get("context_total_tokens"),
        "max_context_call": peak.get("call"),
        "max_context_total_tokens": peak.get("context_total_tokens"),
        "token_usage_by_call": responses,
        "context_measurement": "API-reported input plus output of one request; last context is not cumulative usage. "
        "Includes cached input and any reported image/reasoning tokens. Excludes tool results appended after the "
        "response. Failed requests without usage are unknown, not zero. Reasoning tokens are a subset of output.",
    }


def supervision_accounting(directory):
    """Separate decision overhead, bounded handoffs and physical ownership; never count alerts as errors."""
    state_path = directory / "agent/tools-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    contracts = list(state.get("delegations", {}).values())
    finished = [c for c in contracts if c.get("finished")]
    productive = [c for c in finished if c["steps"] > 0]
    zero = [c for c in finished if c["steps"] == 0]
    tools = rows(directory / "agent/tools.jsonl")
    reasons = Counter(reason for c in finished for reason in set(c.get("wake_reasons", [])))
    zero_reasons = Counter(reason for c in zero for reason in set(c.get("wake_reasons", [])))
    return {
        "astra_tool_dispatches": len(tools),
        "astra_tool_counts": dict(Counter(r["name"] for r in tools)),
        "astra_release_prefix_counts": dict(Counter(str(r["args"]["prefix"]) for r in tools if r["name"] == "release")),
        "delegation_contracts": len(contracts),
        "delegation_finished_contracts": len(finished),
        "delegation_productive_contracts": len(productive),
        "delegation_zero_step_contracts": len(zero),
        "delegation_average_chunks_per_productive_contract": sum(c["chunks"] for c in productive) / len(productive)
        if productive
        else None,
        "delegation_average_steps_per_productive_contract": sum(c["steps"] for c in productive) / len(productive)
        if productive
        else None,
        "delegation_wake_counts": dict(reasons),
        "delegation_zero_step_wake_counts": dict(zero_reasons),
        "delegation_details": [
            {
                "id": c.get("id"),
                "finished": c.get("finished", False),
                "initial_review": c.get("initial_review", "legacy_jev_precheck"),
                "phase": c.get("phase", c.get("args", {}).get("phase")),
                "budget_unit": "controls"
                if c.get("kind") == "step-budget"
                else "legacy_chunks_and_steps"
                if "checkpoint_steps" in c.get("args", {})
                else "chunks",
                "max_chunks": c.get("args", {}).get("max_chunks"),
                "chunk_prefix_limit": c.get("chunk_prefix_limit"),
                "observation_interval_chunks": c.get("observation_interval_chunks", 1),
                "prefix_steps": c.get("args", {}).get("prefix_steps"),
                "checkpoint_steps": c.get("args", {}).get("checkpoint_steps"),
                "executed_chunks": c["chunks"],
                "executed_steps": c["steps"],
                "wake_reasons": c.get("wake_reasons", []),
            }
            for c in contracts
        ],
    }


def execution_accounting(directory, expected_steps):
    """Count actual controller returns; delegated VLA is a subset, never an additional actor."""
    controls = [r for r in rows(directory / "events.jsonl") if r["event"] == "control_return"]
    adviser = rows(directory / "agent/jev-observations.jsonl")
    confidence_reasons = Counter()
    label_reviews = []
    for advice in adviser:
        specification = advice.get("question_spec", {})
        label_reviews.append(
            advice.get("label_review_required", advice.get("label") in specification.get("review_labels", []))
        )
        if advice.get("low_confidence"):
            reasons = advice.get("low_confidence_reasons")
            if reasons is None:
                reasons = confidence_diagnostics(
                    advice.get("probabilities"), advice.get("abstain_probability"), specification
                )["low_confidence_reasons"]
            confidence_reasons.update(reasons or ["unrecorded_policy"])
    state_path = directory / "agent/tools-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    contracts = list(state.get("delegations", {}).values())
    vla = sum(r.get("kind") == "joint" for r in controls)
    gpt = sum(r.get("kind") == "eef" for r in controls)
    delegated = sum(c["steps"] for c in contracts)
    valid = len(controls) == expected_steps and vla + gpt == len(controls) and 0 <= delegated <= vla
    requests = {}
    for control in controls:
        request = control.get("request_id")
        if request is not None:
            group = requests.setdefault(request, {"steps": 0, "first_frame": control.get("before_frame")})
            group["steps"] += 1
    opinions = {}
    for advice in adviser:
        key = (advice.get("contract_id"), advice.get("frame_id"))
        if None not in key:
            opinions.setdefault(key, []).append(advice)
    entry_steps = advised_steps = advised_chunks = counted_delegated = 0
    for contract in contracts:
        identifier = contract.get("id")
        if not isinstance(identifier, str):
            continue
        for request, group in requests.items():
            if not request.startswith(identifier + "-c"):
                continue
            counted_delegated += group["steps"]
            if request == identifier + "-c000" and contract.get("initial_review") == "astra":
                entry_steps += group["steps"]
            matching = opinions.get((identifier, group["first_frame"]), [])
            if matching and all(r["status"] == "answered" and r.get("review_required") is False for r in matching):
                advised_steps += group["steps"]
                advised_chunks += 1
    attribution = valid and counted_delegated == delegated and all("request_id" in r for r in controls)
    return {
        "control_ledger_matches_result": valid,
        "gpt_control_steps": gpt,
        "vla_control_steps": vla,
        "vla_control_requests": len({r["request_id"] for r in controls if r.get("kind") == "joint"})
        if attribution
        else None,
        "unknown_control_steps": len(controls) - vla - gpt,
        "delegated_chunks": sum(c["chunks"] for c in contracts),
        "delegated_steps": delegated,
        "astra_direct_vla_steps": vla - delegated if valid else None,
        "gpt_control_fraction": gpt / expected_steps if valid and expected_steps else None,
        "vla_control_fraction": vla / expected_steps if valid and expected_steps else None,
        "delegated_step_fraction": delegated / expected_steps if valid and expected_steps else None,
        "astra_direct_vla_fraction": (vla - delegated) / expected_steps if valid and expected_steps else None,
        "delegation_review_attribution_available": attribution,
        "astra_entry_delegated_steps": entry_steps if attribution else None,
        "jev_advice_backed_delegated_steps": advised_steps if attribution else None,
        "jev_advice_backed_delegated_chunks": advised_chunks if attribution else None,
        "jev_advice_backed_control_fraction": advised_steps / expected_steps
        if attribution and expected_steps
        else None,
        "delegated_without_current_jev_review_steps": delegated - advised_steps if attribution else None,
        "jev_normal_answers": sum(r["status"] == "answered" and r.get("review_required") is False for r in adviser),
        "jev_alert_answers": sum(
            r["status"] == "answered" and r.get("review_required") is True and not r.get("low_confidence")
            for r in adviser
        ),
        "jev_low_confidence_answers": sum(r["status"] == "answered" and bool(r.get("low_confidence")) for r in adviser),
        "jev_low_confidence_reason_counts": dict(confidence_reasons),
        "jev_label_review_answers": sum(label_reviews),
        "jev_pure_low_confidence_answers": sum(
            r.get("low_confidence", False) and not label_review
            for r, label_review in zip(adviser, label_reviews, strict=True)
        ),
        "jev_low_confidence_with_review_label_answers": sum(
            r.get("low_confidence", False) and label_review
            for r, label_review in zip(adviser, label_reviews, strict=True)
        ),
        "jev_unavailable_answers": sum(r["status"] == "unavailable" for r in adviser),
        "jev_budget_reports": sum(r["status"] == "budget_exhausted" for r in adviser),
    }


def astra_call_substitution(api_calls, operations):
    """Fixed-trajectory review substitution, not an unobserved Astra-only episode prediction."""
    replaced = (
        operations.get("jev_advice_backed_delegated_chunks")
        if operations.get("delegation_review_attribution_available")
        else None
    )
    total = api_calls + replaced if type(api_calls) is int and replaced is not None else None
    vla_chunks = operations.get("vla_control_requests")
    return {
        "estimated_astra_calls_without_jev": total,
        "estimated_astra_calls_replaced_by_jev": replaced,
        "estimated_astra_call_replacement_fraction": replaced / total if total else None,
        "astra_call_fraction_with_jev": api_calls / total if total else None,
        "astra_reviewed_vla_chunks": vla_chunks - replaced if vla_chunks is not None and replaced is not None else None,
        "jev_reviewed_vla_chunk_fraction": replaced / vla_chunks if vla_chunks and replaced is not None else None,
        "astra_call_substitution_method": (
            "Hold the observed controls and all other API/tool interactions fixed; replace each actually executed "
            "JEV-normal-backed VLA chunk with one Astra review. Excludes entry chunks, alerts, unexecuted advice "
            "and duplicate questions. This is review-equivalent accounting, not a measured Astra-only trajectory "
            "or an estimate of saved tokens. API retries remain in the observed API count."
        ),
    }


def validate_frozen(tools, args):
    if not getattr(args, "frozen_manifest", None):
        return None
    path = args.frozen_manifest
    content = path.read_bytes()
    frozen = json.loads(content)
    task = tools.manifest["task"]
    if tools.manifest["experience_update"]:
        raise ValueError("frozen runs forbid experience commits")
    for key, name in (("experience_input", "global"), ("task_experience_input", task)):
        if tools.manifest.get(key, {}).get("sha256") != frozen["experience_sha256"][name]:
            raise ValueError("frozen experience identity mismatch")
    expected = frozen["settings"]
    actual = {
        "model": args.model,
        "reasoning": args.reasoning,
        "max_delegated_chunks": args.max_delegated_chunks,
        "prefix": args.prefix,
        "max_steps": args.max_steps,
        "max_calls": args.max_calls,
        "jev_config_sha256": tools.jev.fingerprint if tools.jev else None,
    }
    if actual != expected:
        raise ValueError("frozen model/policy settings mismatch")
    service = tools.manifest.get("policy", {})
    if service.get("checkpoint") != frozen["checkpoint"]:
        raise ValueError("frozen VLA checkpoint mismatch")
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(content).hexdigest()}


def summarize(tools):
    result = tools.status()["result"]
    config_path = tools.agent_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    api = rows(tools.agent_dir / "api-timing.jsonl")
    completed = [r for r in api if r["event"] == "end"]
    started = [r for r in api if r["event"] == "start"]
    events = rows(tools.directory / "events.jsonl")
    contracts = list(tools.state.get("delegations", {}).values())
    wakes = {}
    for contract in contracts:
        for reason in contract.get("wake_reasons", []):
            wakes[reason] = wakes.get(reason, 0) + 1
    ready, ended = result.get("ready_epoch"), result.get("physical_ended_epoch")
    elapsed = max(0, ended - ready) if ready is not None and ended is not None else None
    started_at = tools.manifest.get("started_at")
    launch_epoch = datetime.fromisoformat(started_at).timestamp() if started_at else None
    physical_closed = result.get("ended_at")
    closed_epoch = datetime.fromisoformat(physical_closed).timestamp() if physical_closed else None
    total_steps = result.get("control_steps", 0)
    delegated = sum(c["steps"] for c in contracts)
    operations = execution_accounting(tools.directory, total_steps)
    return {
        "schema_version": 1,
        "supervision_mode": tools.supervision_mode,
        "max_context_tokens": config.get("max_context_tokens"),
        "frozen": tools.state.get("frozen_manifest"),
        "episode_wall_seconds": elapsed,
        "startup_seconds": max(0, ready - launch_epoch) if ready is not None and launch_epoch is not None else None,
        "cleanup_seconds": max(0, closed_epoch - ended) if closed_epoch is not None and ended is not None else None,
        "run_wall_seconds": max(0, time.time() - launch_epoch) if launch_epoch is not None else None,
        "completion_seconds": elapsed if result.get("feedback", {}).get("success") is True else None,
        "api_attempts_logged": len(started),
        "api_successes": sum(r["ok"] for r in completed),
        "api_failures": sum(not r["ok"] for r in completed),
        "api_unresolved_attempts": len(started) - len(completed),
        "api_retries": sum(r["attempt"] > 0 for r in started),
        "api_seconds": sum(r["seconds"] for r in completed),
        "api_execution_calls": sum(r["phase"] == "execution" for r in started),
        "api_review_calls": sum(r["phase"] == "review" for r in started),
        "cached_input_tokens": sum(r.get("cached_input_tokens", 0) for r in completed),
        "jev": dict(tools.state["jev_accounting"]),
        "jev_identity": {key: tools.jev.describe()[key] for key in ("model", "revision", "protocol", "config_sha256")}
        if tools.jev
        else None,
        "vla_inference_seconds": sum(r.get("inference_seconds", 0) for r in events if r["event"] == "candidate"),
        "control_seconds": sum(r["seconds"] for r in events if r["event"] == "control_return"),
        "delegated_steps": delegated,
        "delegated_step_fraction": delegated / total_steps if total_steps else 0,
        "wake_counts": wakes,
        "experience": {k: tools.manifest.get(k) for k in ("experience_input", "task_experience_input")},
        "event_detection_annotations": [],
        "annotation_note": "Unlabelled, not zero errors. Label visible evidence and wake frames during review.",
        "execution_accounting": operations,
        "astra_call_substitution": astra_call_substitution(len(started), operations),
        "supervision_accounting": supervision_accounting(tools.directory),
        "token_accounting": token_accounting(tools.directory),
        "scheduling_accounting": scheduling_accounting(tools.directory)
        if tools.supervision_mode == "step-budget"
        else None,
    }


def scheduling_accounting(directory):
    """Actual step-budget costs and fixed-trajectory review equivalents, never counterfactual tokens."""
    state_path = directory / "agent/tools-state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    contracts = [c for c in state.get("delegations", {}).values() if c.get("kind") == "step-budget"]
    controls = [r for r in rows(directory / "events.jsonl") if r.get("event") == "control_return"]
    requests = Counter(r.get("request_id") for r in controls if r.get("kind") == "joint")
    matched = {
        c["id"]: [key for key in requests if isinstance(key, str) and key.startswith(c["id"] + "-c")] for c in contracts
    }
    valid = all(
        sum(requests[key] for key in matched[c["id"]]) == c["steps"] and len(matched[c["id"]]) == c["chunks"]
        for c in contracts
    )
    equivalents = sum(max(0, len(matched[c["id"]]) - 1) for c in contracts) if valid else None
    timings = rows(directory / "agent/api-timing.jsonl")
    starts = [r for r in timings if r.get("event") == "start"]
    phases = {r["call"]: r.get("phase") for r in starts}
    responses = {}
    categories = Counter()
    for path in sorted((directory / "agent").glob("response-*.json")):
        data = json.loads(path.read_text())
        call = int(path.stem.removeprefix("response-"))
        names = [o["name"] for o in data.get("output", []) if o.get("type") == "function_call"]
        if phases.get(call) == "review":
            category = "review"
        elif any(n in ("history", "history_sample", "view_image") for n in names):
            category = "history_request"
        elif any(n in ("schedule_vla", "release", "intervene", "set_subgoal") for n in names):
            category = "decision"
        elif "write_memory" in names:
            category = "memory"
        else:
            category = "other"
        categories[category] += 1
        responses[call] = category
    usage = token_accounting(directory)
    costs = {}
    for category in ("execution", "review"):
        selected = [
            r for r in usage["token_usage_by_call"] if (phases.get(r["call"]) == "review") == (category == "review")
        ]
        costs[category] = {
            key: sum(r[key] for r in selected) if selected and all(r[key] is not None for r in selected) else None
            for key in (
                "input_tokens",
                "cached_input_tokens",
                "uncached_input_tokens",
                "output_tokens",
                "context_total_tokens",
            )
        }
    return {
        "contract_count": len(contracts),
        "requested_steps": sum(c["args"]["budget_steps"] for c in contracts),
        "executed_steps": sum(c["steps"] for c in contracts),
        "zero_step_reports": sum(c["steps"] == 0 and c.get("finished", False) for c in contracts),
        "guard_trigger_counts": dict(Counter(e["reason"] for c in contracts for e in c.get("guard_events", []))),
        "contracts": [
            {
                "id": c["id"],
                "subgoal": c["phase"],
                "budget_steps": c["args"]["budget_steps"],
                "executed_steps": c["steps"],
                "remaining_steps": c["args"]["budget_steps"] - c["steps"],
                "chunks": c["chunks"],
                "wake_reasons": c.get("wake_reasons", []),
                "guard_events": c["guard_events"],
            }
            for c in contracts
        ],
        "execution_ledger_matches_contracts": valid,
        "avoided_chunk_review_equivalents": equivalents,
        "fixed_trajectory_calls_with_chunk_reviews": len(starts) + equivalents if equivalents is not None else None,
        "review_equivalent_reduction_fraction": equivalents / (len(starts) + equivalents)
        if equivalents is not None and len(starts) + equivalents
        else None,
        "estimate_note": "Each actually executed chunk after a contract's Astra-reviewed entry counts as one "
        "avoided per-chunk review. No zero-step/unexecuted request counts. This fixes the observed trajectory; "
        "it is not an independent baseline, a causal effect, or saved-token estimate.",
        "response_categories": dict(categories),
        "calls_without_response": len(starts) - len(responses),
        "retry_attempts": sum(r.get("attempt", 0) > 0 for r in starts),
        "tokens_by_phase": costs,
        "astra_delivered_frame_count": len(state.get("seen", [])),
    }
