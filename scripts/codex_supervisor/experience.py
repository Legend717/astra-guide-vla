"""Append-only evidence entries, independent read/write gates and immutable run snapshot."""

import fcntl
import hashlib
import json
from pathlib import Path
import re

from scripts.codex_rgb_eval import read_json
from scripts.codex_rgb_eval import write_json


def snapshot(directory: Path, source: Path, *, enabled: bool, filename="experience-input.json") -> dict:
    metadata = {"enabled": enabled, "source": str(source.resolve()), "sha256": None}
    if enabled:
        content = source.read_bytes() if source.exists() else b'{"version": 0, "entries": []}\n'
        target = directory / filename
        target.write_bytes(content)
        target.chmod(0o444)
        metadata["sha256"] = hashlib.sha256(content).hexdigest()
    return metadata


def applicability(entry):
    """Explicit labels win; conservative legacy fallback examines content, not run provenance."""
    if entry.get("applicability"):
        return entry["applicability"]
    if "scheduling" in entry:
        return "step-budget"
    if "supervision" in entry or re.search(
        r"jev|omnijev|review_labels|supervise_vla|置信.*门槛",
        entry.get("claim", "") + " " + entry.get("conditions", ""),
        re.IGNORECASE,
    ):
        return "jev"
    return "common"


def active_view(snapshot_data, *, mode=None):
    """Filter modes before supersession, preserving the immutable source and delivery provenance."""
    entries = snapshot_data["entries"]
    eligible = (
        entries
        if mode is None
        else [
            e
            for e in entries
            if applicability(e)
            in (
                {"common", "jev"}
                if mode == "jev"
                else {"common", "step-budget"}
                if mode == "step-budget"
                else {"common"}
            )
        ]
    )
    superseded = {identifier for entry in eligible for identifier in entry.get("supersedes", [])}
    active = [e for e in eligible if e.get("id") not in superseded and e.get("status", "active") == "active"]
    return {**snapshot_data, "entries": active, "inactive_entry_ids": [e.get("id") for e in entries if e not in active]}


def update_policy(manifest: dict, result: dict) -> dict:
    """Resolve the immutable run policy against the official outcome, never a model's claim."""
    policy = manifest.get("experience_update_policy", "always")
    if policy not in ("always", "failure-only"):
        raise ValueError("unknown experience update policy")
    if not manifest["experience_update"]:
        reason = "experience updates are disabled for this run"
    elif result.get("lifecycle") != "closed":
        reason = "experience may only be committed after the episode closes"
    elif policy == "failure-only" and (
        result.get("outcome") == "success" or result.get("feedback", {}).get("success") is True
    ):
        reason = "failure-only experience updates are disabled after official success"
    elif policy == "failure-only" and result.get("outcome") not in ("failure", "timeout", "error", "incomplete"):
        reason = "failure-only experience updates require a known unsuccessful outcome"
    else:
        reason = None
    return {"enabled": manifest["experience_update"], "policy": policy, "allowed": reason is None, "reason": reason}


def commit(directory: Path, submission: dict) -> dict:
    manifest = read_json(directory / "manifest.json")
    result = read_json(directory / "result.json")
    gate = update_policy(manifest, result)
    if not gate["allowed"]:
        raise ValueError(gate["reason"])
    entries = submission.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("entries must be a nonempty list")
    required = {"claim", "conditions", "confidence", "frames", "kind"}
    allowed = required | {"phase", "supervision", "scheduling", "applicability", "supersedes", "status"}
    for entry in entries:
        if not isinstance(entry, dict) or required - set(entry) or set(entry) - allowed:
            raise ValueError("invalid experience entry fields")
        if any(not isinstance(entry[k], str) or not entry[k].strip() for k in ("claim", "conditions")):
            raise ValueError("claim and conditions must be nonempty strings")
        if entry["confidence"] not in ("low", "medium", "high"):
            raise ValueError("confidence must be low, medium or high")
        if entry["kind"] not in ("observation", "hypothesis", "engineering"):
            raise ValueError("kind must distinguish observation, hypothesis or engineering")
        if entry.get("status", "active") not in ("active", "contradicted", "disabled"):
            raise ValueError("invalid experience status")
        if "phase" in entry and (not isinstance(entry["phase"], str) or not entry["phase"].strip()):
            raise ValueError("phase must be nonempty")
        if "supersedes" in entry and (
            not isinstance(entry["supersedes"], list) or any(not isinstance(x, str) for x in entry["supersedes"])
        ):
            raise ValueError("supersedes requires entry IDs")
        if entry.get("applicability", "common") not in ("common", "jev", "step-budget"):
            raise ValueError("invalid experience applicability")
        if "scheduling" in entry:
            if submission.get("scope", "global") != "task":
                raise ValueError("subgoal scheduling experience requires task scope")
            from scripts.codex_supervisor.instructions import TEMPLATES
            from scripts.codex_supervisor.instructions import matches_template
            from scripts.robomme_agent.scheduling import SCHEDULING_EXPERIENCE
            from scripts.robomme_agent.scheduling import validate_guards
            from scripts.robomme_agent.tools import validate_schema

            advice = entry["scheduling"]
            validate_schema(advice, SCHEDULING_EXPERIENCE)
            validate_guards(advice.get("guards", []))
            if not any(
                matches_template(advice["subgoal"], template)
                for template in TEMPLATES.get(manifest.get("task"), {}).get("simple", [])
            ):
                raise ValueError("scheduling subgoal must match this task's simple template")
            if entry.get("applicability", "step-budget") != "step-budget":
                raise ValueError("scheduling experience requires step-budget applicability")
            if "first_risk_frame" in advice and advice["first_risk_frame"] not in entry["frames"]:
                raise ValueError("first risk frame must be cited as evidence")
        if "supervision" in entry:
            # Share the public boundary validation; avoid divergent strict schemas.
            from scripts.robomme_agent.tools import TOOLS
            from scripts.robomme_agent.tools import validate_schema

            schema = next(t for t in TOOLS if t["name"] == "experience_commit")
            validate_schema(
                entry["supervision"],
                schema["parameters"]["properties"]["entries"]["items"]["properties"]["supervision"],
            )
        if (manifest["mode"] == "simple-smoke" or manifest.get("policy_backend", "real") != "real") and entry[
            "kind"
        ] != "engineering":
            raise ValueError("smoke runs may only contribute engineering evidence, not task-success experience")
        frames = entry["frames"]
        if (
            not isinstance(frames, list)
            or not frames
            or any(type(f) is not int or not 0 <= f < result["recorded_frames"] for f in frames)
        ):
            raise ValueError("frames must reference this run's archived observations")
    scope = submission.get("scope", "global")
    if scope not in ("global", "task"):
        raise ValueError("invalid experience scope")
    source_key = "task_experience_input" if scope == "task" else "experience_input"
    if source_key not in manifest:
        raise ValueError("task experience is not configured for this run")
    payload = {"run": str(directory.resolve()), "entries": entries, "scope": scope}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    target = Path(manifest[source_key]["source"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix(target.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = read_json(target) if target.exists() else {"version": 0, "entries": [], "commits": []}
        known_ids = {e.get("id") for e in previous["entries"]}
        if any(set(e.get("supersedes", [])) - known_ids for e in entries):
            raise ValueError("supersedes references an unknown experience ID")
        if digest not in previous.get("commits", []):
            provenance = {
                "task": manifest.get("task"),
                "supervision_mode": (
                    read_json(directory / "agent/tools-state.json").get("supervision_config", {})
                    if (directory / "agent/tools-state.json").exists()
                    else {}
                ),
                "checkpoint": manifest.get("checkpoint") or manifest.get("policy", {}).get("checkpoint"),
            }
            config_path = directory / "agent/jev-config.json"
            if config_path.exists():
                config = read_json(config_path)
                provenance.update(jev_model=config["model"], jev_revision=config["revision"])
                provenance["jev_config_sha256"] = hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode()
                ).hexdigest()
            updated = {
                **previous,
                "schema_version": 2,
                "version": previous["version"] + 1,
                "entries": previous["entries"]
                + [
                    {**entry, "id": f"{digest}:{i}", "source_run": payload["run"], "provenance": provenance}
                    for i, entry in enumerate(entries)
                ],
                "commits": [*previous.get("commits", []), digest],
            }
            versions = target.parent / (target.name + ".versions")
            versions.mkdir(exist_ok=True)
            write_json(versions / f"{updated['version']:07d}-{digest}.json", updated)
            write_json(target, updated)
        else:
            updated = previous
    receipt = {"scope": scope, "commit_sha256": digest, "version": updated["version"], "added_entries": len(entries)}
    write_json(directory / ("experience-task-commit.json" if scope == "task" else "experience-commit.json"), receipt)
    return receipt  # Never return old experience text when input is disabled.


def migrate(source: Path, destination: Path) -> dict:
    """Create new scoped libraries without modifying the historical source or its archives."""
    if destination.exists():
        raise ValueError("migration destination must be new")
    previous = read_json(source)
    libraries = {key: [] for key in ("global", "InsertPeg", "StopCube")}
    mapping = []
    for index, original in enumerate(previous["entries"]):
        claim = original["claim"] + " " + original["conditions"]
        matched = [task for task in ("InsertPeg", "StopCube") if task.lower() in claim.lower()]
        run = Path(original.get("source_run", "/nonexistent"))
        if not matched and (run / "manifest.json").exists():
            task = read_json(run / "manifest.json").get("task")
            if task in libraries and task != "global":
                matched = [task]
        scope = matched[0] if len(matched) == 1 else "global"
        identifier = original.get("id", f"legacy-v{previous['version']}-{index}")
        libraries[scope].append({**original, "id": identifier})
        mapping.append({"source_index": index, "id": identifier, "scope": scope})
    destination.mkdir(parents=True)
    for scope, entries in libraries.items():
        write_json(
            destination / f"{scope}.json", {"schema_version": 2, "version": 1, "entries": entries, "commits": []}
        )
    result = {
        "source": str(source.resolve()),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mapping": mapping,
    }
    write_json(destination / "migration.json", result)
    return result
