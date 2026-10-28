"""Durable bounded delegation. Model opinions never create or extend execution authority."""

from __future__ import annotations

import hashlib
import math
import re

from PIL import Image
from PIL import ImageDraw

from scripts import codex_rgb_eval as rgb
from scripts.robomme_agent.jev import validate_question

QUESTION = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "purpose": {"type": "string", "enum": ["visual_fact", "stage_warning"]},
        "camera": {"type": "string", "enum": ["front", "wrist", "both"]},
        "labels": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 16},
        "review_labels": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    },
    "required": ["prompt", "purpose", "labels", "review_labels"],
    "additionalProperties": False,
}
CONTRACT_PROPERTIES = {
    "phase": {"type": "string"},
    "phase_mode": {"type": "string", "enum": ["ordinary", "critical", "recovery", "timing"]},
    "max_chunks": {"type": "integer", "minimum": 1},
    "max_no_progress_checks": {"type": "integer", "minimum": 1, "maximum": 10},
    "motion_expected": {"type": "boolean"},
    "grasp_state": {"type": "string", "enum": ["none", "acquire", "holding"]},
    "wake_on_gripper_change": {"type": "boolean"},
    "questions": {"type": "array", "items": QUESTION, "minItems": 1, "maxItems": 3},
    "experience_refs": {"type": "array", "items": {"type": "string"}},
    "reason": {"type": "string"},
}


def identity(status):
    d = status["decision"]
    return {
        "expected_frame": d["expected_frame"],
        "subgoal_version": d["subgoal_version"],
        "candidate_id": (d["candidate"] or {}).get("candidate_id"),
    }


def finite_robot(robot):
    return all(
        isinstance(robot.get(key), list)
        and len(robot[key]) == size
        and all(type(x) in (int, float) and math.isfinite(x) for x in robot[key])
        for key, size in (("eef_state", 6), ("joint_state", 7), ("gripper_state", 2))
    )


def default_completion_question(grounded_subgoal):
    # Coordinates ground VLA actions in the original front image, not visual completion judgments.
    # Remove only the converter's location syntax; preserve counts, identity and grasp-side wording.
    visual_subgoal = re.sub(r"\s+at\s+<\d+,\s*\d+>", "", grounded_subgoal)
    return {
        "prompt": "In these current front and wrist views, is this visible subgoal already accomplished? "
        + visual_subgoal
        + " Answer uncertain if the evidence is insufficient; an attempted action is not completion.",
        "purpose": "visual_fact",
        "labels": ["visibly_complete", "not_yet_complete", "uncertain"],
        "review_labels": ["visibly_complete", "uncertain"],
    }


def current_panel(tools, frame, target):
    """Match the evaluated 512x280 latest front/wrist panel; no future or inferred pixels."""
    panel = Image.new("RGB", (512, 280), "white")
    draw = ImageDraw.Draw(panel)
    sources = {}
    for col, camera in enumerate(("front", "wrist")):
        source = tools.image(frame["rgb"][camera])
        sources[camera] = hashlib.sha256(source.read_bytes()).hexdigest()
        with Image.open(source) as im:
            if im.size != (256, 256):
                raise ValueError("delegation requires native 256x256 camera frames")
            panel.paste(im.convert("RGB"), (256 * col, 24))
        draw.text((256 * col + 4, 5), f"frame {frame['frame_id']} | {camera}", fill="black")
    panel.save(target)
    return sources


class Delegator:
    def __init__(self, tools):
        self.tools = tools

    def save(self):
        self.tools.save()

    def finish(self, contract, reasons):
        contract["wake_reasons"] = list(dict.fromkeys(reasons))
        contract["finished"] = True
        self.tools.state["active_delegation"] = None
        self.save()
        return {
            "contract_id": contract["id"],
            "wake_reasons": contract["wake_reasons"],
            "executed_chunks": contract["chunks"],
            "executed_steps": contract["steps"],
            "first_frame": contract["args"]["expected_frame"],
            "last_frame": self.tools.status()["latest"]["frame_id"],
            "advice": contract.get("advice", []),
            "status": self.tools.status(),
        }, []

    def run(self, args, identifier):
        t = self.tools
        contracts = t.state.setdefault("delegations", {})
        if identifier not in contracts:
            if "checkpoint_steps" in args or "prefix_steps" in args:
                raise ValueError("use max_chunks as the only delegation budget; prefix is managed by the harness")
            if t.supervision_mode == "astra-only" or t.state.get("jev_disabled"):
                raise ValueError("delegation unavailable; use Astra review and release")
            if t.state.get("pending") or t.state.get("active_delegation"):
                raise ValueError("resolve the original pending delegation first")
            t.check_seen()
            if t.state.get("memory_checkpoint_required"):
                raise ValueError("save reviewed evidence with write_memory before delegation")
            status = t.status()
            if any(args[key] != value for key, value in identity(status).items()):
                raise ValueError("stale delegation identity")
            if not status["decision"]["candidate"]:
                raise ValueError("set a subgoal before delegation")
            if args["max_chunks"] > t.max_delegated_chunks:
                raise ValueError("delegation exceeds configured hard chunk limit")
            if args["phase_mode"] != "ordinary" and args["max_chunks"] != 1:
                raise ValueError("critical/recovery/timing phases require Astra every chunk")
            questions = args.get("questions") or [
                default_completion_question(status["decision"]["subgoal"]["grounded_subgoal"])
            ]
            for q in questions:
                validate_question(q)
            interval = (
                tconfig(t).get("reporting_policy", {}).get("ordinary_observation_interval_chunks", 1)
                if args["phase_mode"] == "ordinary"
                else 1
            )
            contracts[identifier] = {
                "id": identifier,
                "args": args,
                "chunk_prefix_limit": 4 if args["phase_mode"] == "timing" else 16,
                "observation_interval_chunks": interval,
                "next_observation_chunk": interval,
                "chunks": 0,
                "steps": 0,
                "no_progress": 0,
                "next_request": None,
                "finished": False,
                # Astra has just reviewed this exact frame/candidate and authorized the first prefix.
                # JEV starts on newly executed observations, not a redundant entry-frame query.
                "checked_frame": status["latest"]["frame_id"],
                "initial_review": "astra",
                "advice": [],
                "questions": questions,
            }
            t.state["active_delegation"] = identifier
            self.save()
        c = contracts[identifier]
        if args != c["args"]:
            raise ValueError("contract ID reused with different arguments")
        if c["finished"]:
            return self.finish(c, c["wake_reasons"])
        return self.drive(c)

    def prefix(self, c, status):
        args = c["args"]
        candidate = status["decision"]["candidate"]
        prefix = min(c.get("chunk_prefix_limit", args.get("prefix_steps", 16)), candidate["default_prefix"])
        if "checkpoint_steps" in args:
            prefix = min(prefix, args["checkpoint_steps"] - c["steps"])
        return prefix

    def on_receipt(self, c, child, steps):
        """Extension point for measured-state guards; called before the durable receipt transition."""

    def drive(self, c):
        """Shared exactly-once execution loop; policy-specific budgets and checks stay in adapters."""
        t, args, identifier = self.tools, c["args"], c["id"]
        while True:
            # Resume the exact original child request before inspecting the changed world.
            if c["next_request"]:
                child = c["next_request"]
                if c.get("cancel_requested") and not (t.directory / "requests" / f"{child['id']}.json").exists():
                    # A durable intent is not yet physical work. Only settle requests already enqueued.
                    if t.state.get("pending") == child["id"]:
                        t.state["pending"] = None
                    c["next_request"] = None
                    return self.finish(c, ["delegation_cancelled"])
                result = t.submit("release", child["args"], child["id"])
                if result.get("pending"):
                    return {**result, "contract_id": identifier}, []
                if not result.get("ok"):
                    return self.finish(c, ["physical_request_rejected"])
                after = t.status()
                steps = result.get("executed_steps", 0)
                c["chunks"] += int(steps > 0)
                c["steps"] += steps
                c["last_executed"] = child
                c["next_request"] = None
                before_robot, after_robot = child["robot"], after["latest"]["robot"]
                if finite_robot(before_robot) and finite_robot(after_robot):
                    # A bounded robot-motion check, not an assertion of task-level progress.
                    stagnant = (
                        max(
                            abs(x - y)
                            for x, y in zip(before_robot["joint_state"], after_robot["joint_state"], strict=True)
                        )
                        < 0.001
                        and max(
                            abs(x - y)
                            for x, y in zip(before_robot["gripper_state"], after_robot["gripper_state"], strict=True)
                        )
                        < 0.001
                    )
                    c["no_progress"] = c["no_progress"] + 1 if stagnant else 0
                self.on_receipt(c, child, steps)
                self.save()  # Receipt application and counter update are one durable transition.
                if steps == 0:
                    return self.finish(c, ["zero_execution"])
                if result.get("interrupted") or after["result"].get("feedback", {}).get("control_error"):
                    return self.finish(c, ["control_error"])
            status = t.status()
            reasons = self.hard_reasons(c, status)
            if reasons:
                return self.finish(c, reasons)
            if (
                t.supervision_mode == "jev"
                and c["checked_frame"] != status["latest"]["frame_id"]
                and c["chunks"] >= c.get("next_observation_chunk", 0)
            ):
                reasons = self.observe(c, status)
                if reasons:
                    return self.finish(c, reasons)
                # A late response cannot authorize a changed candidate, stage, or expired budget.
                latest = t.status()
                if identity(latest) != identity(status):
                    return self.finish(c, ["stale_advice"])
                reasons = self.hard_reasons(c, latest)
                if reasons:
                    return self.finish(c, reasons)
            candidate = status["decision"]["candidate"]
            prefix = self.prefix(c, status)
            child_id = f"{identifier}-c{c['chunks']:03d}"
            c["next_request"] = {
                "id": child_id,
                "args": {**identity(status), "prefix": prefix, "reason": f"contract {identifier}: {args['reason']}"},
                "robot": status["latest"]["robot"],
                "trajectory": candidate["eef"]["trajectory"][:prefix],
            }
            self.save()  # Never enqueue physical work before persisting its immutable ID/body.

    def hard_reasons(self, c, status):
        args = c["args"]
        reasons = []
        if c.get("cancel_requested"):
            return ["delegation_cancelled"]
        if status["result"]["lifecycle"] == "closed" or status["result"].get("outcome") != "incomplete":
            return ["episode_ended"]
        if status["decision"]["subgoal_version"] != args["subgoal_version"]:
            reasons.append("subgoal_changed")
        if c["chunks"] >= args["max_chunks"]:
            reasons.append("chunk_budget")
        if "checkpoint_steps" in args and c["steps"] >= args["checkpoint_steps"]:
            reasons.append("phase_checkpoint")
        if args["motion_expected"] and c["no_progress"] >= args["max_no_progress_checks"]:
            reasons.append("no_robot_progress")
        robot = status["latest"]["robot"]
        if not finite_robot(robot):
            reasons.append("invalid_robot_state")
        candidate = status["decision"]["candidate"]
        if candidate is None:
            reasons.append("missing_candidate")
        else:
            trajectory = candidate["eef"]["trajectory"]
            commands = [x.get("gripper_command") for x in trajectory]
            if not commands or any(type(x) not in (int, float) or not math.isfinite(x) for x in commands):
                reasons.append("invalid_candidate_state")
            else:
                previous = c.get("last_executed", {}).get("trajectory", [])
                previous_command = previous[-1]["gripper_command"] if previous else None
                signs = [x > 0 for x in commands[: max(0, self.prefix(c, status))]]
                if (
                    args["wake_on_gripper_change"]
                    and signs
                    and (len(set(signs)) > 1 or (previous_command is not None and signs[0] != (previous_command > 0)))
                ):
                    reasons.append("candidate_gripper_transition")
        executed = c.get("last_executed", {}).get("trajectory", [])
        if executed:
            closed = any(x.get("gripper_command", 1) < 0 for x in executed)
            if args["grasp_state"] in ("acquire", "holding") and closed:
                # Opt-in only after task/units verification; never applies to button pressing.
                threshold = tconfig(self.tools).get("empty_grasp_threshold_m")
                if threshold is not None and finite_robot(robot) and sum(robot["gripper_state"]) <= threshold:
                    reasons.append("possible_empty_grasp")
                before_z = c["last_executed"]["robot"]["eef_state"][2]
                if args["grasp_state"] == "acquire" and any(
                    x["target_xyz_rpy"][2] - before_z >= 0.005 for x in executed
                ):
                    reasons.append("grasp_lift_checkpoint")
            if args["grasp_state"] == "holding" and any(x.get("gripper_command", -1) > 0 for x in executed):
                reasons.append("release_verification")
        return reasons

    def observe(self, c, status):
        t = self.tools
        frame = status["latest"]
        panel_path = t.agent_dir / f"{c['id']}-frame-{frame['frame_id']}.png"
        try:
            sources = current_panel(t, frame, panel_path)
        except (OSError, ValueError):
            t.state["jev_disabled"] = True
            self.save()
            return ["invalid_visual_input"]
        reasons = []
        for index, specification in enumerate(c["questions"]):
            # Persist started state: a crash during HTTP returns to Astra, never silently retries.
            marker = f"{frame['frame_id']}:{index}"
            if c.get("pending_observation"):
                t.state["jev_disabled"] = True
                return ["interrupted_advice"]
            c["pending_observation"] = marker
            self.save()
            q = {**tconfig(t).get("probability_policy", {}), **specification}
            camera = q.get("camera", "both")
            image_path = panel_path
            image_sources = sources
            if camera != "both":
                image_path = panel_path.with_name(f"{panel_path.stem}-{camera}.png")
                left = 0 if camera == "front" else 256
                with Image.open(panel_path) as panel:
                    panel.crop((left, 0, left + 256, 280)).save(image_path)
                image_sources = {camera: sources[camera]}
            result = t.jev.observe(
                f"{c['id']}:{index}", [image_path], t.state["jev_accounting"], on_charge=t.save, specification=q
            )
            result.update(
                contract_id=c["id"],
                frame_id=frame["frame_id"],
                candidate_id=identity(status)["candidate_id"],
                camera=camera,
                source_sha256=image_sources,
                image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            )
            rgb.append_json(t.agent_dir / "jev-observations.jsonl", result)
            c["pending_observation"] = None
            c["advice"].append(result)
            if result["status"] == "answered":
                t.state["jev_delivered_frames"] = sorted(
                    set(t.state.get("jev_delivered_frames", [])) | {frame["frame_id"]}
                )
            if result["status"] != "answered":
                t.state["jev_disabled"] = True
                reasons.append("jev_" + result["status"])
            elif result["review_required"]:
                reasons.append("jev_low_confidence" if result.get("low_confidence") else "jev_alert")
            self.save()
            if reasons:
                break
        c["checked_frame"] = frame["frame_id"]
        c["next_observation_chunk"] = c["chunks"] + c.get("observation_interval_chunks", 1)
        self.save()
        return reasons


def tconfig(tools):
    return tools.jev.config if tools.jev else tools.supervision_config
