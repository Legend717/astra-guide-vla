"""Step-budget authorization and optional robot-only guards. No model observer is involved."""

from __future__ import annotations

import json
import math
import re

from scripts.robomme_agent.supervision import Delegator
from scripts.robomme_agent.supervision import finite_robot
from scripts.robomme_agent.supervision import identity

RULES = {
    "version": "robot-guards-v1",
    "direction_tolerance_m": 0.02,
    "gripper_delta_m": 0.005,
    "gripper_debounce_samples": 2,
    "stationary_controls": 32,
    "stationary_position_m": 0.002,
    "stationary_angle_rad": math.radians(2),
}
GUARD = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["gripper_change", "forbidden_direction", "stationary"]},
        "when": {"type": "string", "enum": ["before_command", "after_measured"]},
        "direction": {"type": "string", "enum": ["+x", "-x", "+y", "-y", "+z", "-z"]},
    },
    "required": ["kind"],
    "additionalProperties": False,
}
PROPERTIES = {
    "budget_steps": {"type": "integer", "minimum": 30, "multipleOf": 10},
    "guards": {"type": "array", "items": GUARD, "maxItems": 8},
    "experience_refs": {"type": "array", "items": {"type": "string"}},
    "reason": {"type": "string"},
}
SCHEDULING_EXPERIENCE = {
    "type": "object",
    "properties": {
        "subgoal": {"type": "string"},
        "recommended_steps": {"type": "integer", "minimum": 10, "multipleOf": 10},
        "max_trust_steps": {"type": "integer", "minimum": 1},
        "close_review": {"type": "boolean"},
        "guards": {"type": "array", "items": GUARD, "maxItems": 8},
        "first_risk_frame": {"type": "integer", "minimum": 0},
        "risk_offset_steps": {"type": "integer", "minimum": 0},
        "evidence_count": {"type": "integer", "minimum": 1},
    },
    "required": ["subgoal"],
    "additionalProperties": False,
}


def subgoal_name(status):
    subgoal = status["decision"].get("subgoal") or {}
    return subgoal.get("simple_subgoal") or re.sub(r"\s+at\s+<\d+,\s*\d+>", "", subgoal.get("grounded_subgoal", ""))


def validate_guards(guards):
    for guard in guards:
        kind = guard["kind"]
        allowed = {
            "gripper_change": {"kind", "when"},
            "forbidden_direction": {"kind", "direction"},
            "stationary": {"kind"},
        }[kind]
        if set(guard) - allowed or (kind == "forbidden_direction" and "direction" not in guard):
            raise ValueError("guard fields do not match its kind")
    normalized = [
        {**g, "when": g.get("when", "after_measured")} if g["kind"] == "gripper_change" else g for g in guards
    ]
    if len({json.dumps(g, sort_keys=True) for g in normalized}) != len(guards):
        raise ValueError("duplicate guards")


def orientation_distance(a, b):
    # Extrinsic xyz RPY to quaternion, avoiding angle-wrap and Euler singularities.
    def quaternion(rpy):
        r, p, y = (v / 2 for v in rpy)
        cr, cp, cy = math.cos(r), math.cos(p), math.cos(y)
        sr, sp, sy = math.sin(r), math.sin(p), math.sin(y)
        return (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )

    dot = abs(sum(x * y for x, y in zip(quaternion(a), quaternion(b), strict=True)))
    return 2 * math.acos(min(1.0, dot))


class StepScheduler(Delegator):
    def run(self, args, identifier):
        t = self.tools
        if t.supervision_mode != "step-budget":
            raise ValueError("schedule_vla requires step-budget mode")
        contracts = t.state.setdefault("delegations", {})
        if identifier not in contracts:
            budget = args["budget_steps"]
            if type(budget) is not int or budget < 30 or budget % 10 or budget > t.max_scheduled_steps:
                raise ValueError("budget_steps must be a multiple of ten between 30 and the configured maximum")
            guards = args.get("guards", [])
            validate_guards(guards)
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
            available = {e.get("id") for e in t.experience_entries()}
            if set(args.get("experience_refs", [])) - available:
                raise ValueError("experience_refs must reference active loaded experience")
            robot = status["latest"]["robot"]
            contracts[identifier] = {
                "id": identifier,
                "kind": "step-budget",
                "args": args,
                "phase": subgoal_name(status),
                "rules": dict(RULES),
                "anchor_robot": robot,
                "chunks": 0,
                "steps": 0,
                "no_progress": 0,
                "next_request": None,
                "finished": False,
                "initial_review": "astra",
                "last_command": t.state.get("last_gripper_command"),
                "samples": [{"frame": status["latest"]["frame_id"], "pose": robot["eef_state"]}],
                "guard_events": [],
                "gripper_streak": 0,
                "gripper_sign": 0,
            }
            t.state["active_delegation"] = identifier
            self.save()
        c = contracts[identifier]
        if c["args"] != args or c.get("kind") != "step-budget":
            raise ValueError("contract ID reused with different arguments")
        if c["finished"]:
            return self.finish(c, c["wake_reasons"])
        return self.drive(c)

    def prefix(self, c, status):
        return min(16, status["decision"]["candidate"]["default_prefix"], c["args"]["budget_steps"] - c["steps"])

    def finish(self, contract, reasons):
        result, images = super().finish(contract, reasons)
        result.update(
            budget_steps=contract["args"]["budget_steps"],
            remaining_steps=max(0, contract["args"]["budget_steps"] - contract["steps"]),
            guard_events=contract["guard_events"],
        )
        return result, images

    def event(self, c, reason, frame, **measurements):
        if not any(e["reason"] == reason for e in c["guard_events"]):
            c["guard_events"].append({"reason": reason, "frame": frame, **measurements})

    def on_receipt(self, c, child, steps):
        previous = child["args"]["expected_frame"]
        last = self.tools.status()["latest"]["frame_id"]
        samples = [json.loads(line) for line in (self.tools.directory / "frames.jsonl").read_text().splitlines()]
        samples = [s for s in samples if previous < s["frame_id"] <= last]
        rules = c["rules"]
        for sample in samples:
            robot, frame = sample["robot"], sample["frame_id"]
            if not finite_robot(robot):
                self.event(c, "invalid_robot_state", frame)
                continue
            c["samples"].append({"frame": frame, "pose": robot["eef_state"]})
            c["samples"] = c["samples"][-(rules["stationary_controls"] + 1) :]
            for guard in c["args"].get("guards", []):
                if guard["kind"] == "gripper_change" and guard.get("when", "after_measured") == "after_measured":
                    delta = sum(robot["gripper_state"]) - sum(c["anchor_robot"]["gripper_state"])
                    sign = (1 if delta > 0 else -1) if abs(delta) >= rules["gripper_delta_m"] else 0
                    c["gripper_streak"] = (
                        c["gripper_streak"] + 1 if sign and sign == c["gripper_sign"] else int(bool(sign))
                    )
                    c["gripper_sign"] = sign
                    if c["gripper_streak"] >= rules["gripper_debounce_samples"]:
                        self.event(c, "measured_gripper_change", frame, delta_m=delta)
                elif guard["kind"] == "forbidden_direction":
                    self.check_direction(c, guard, robot["eef_state"], frame, "measured_direction_limit")
                elif guard["kind"] == "stationary" and len(c["samples"]) >= rules["stationary_controls"] + 1:
                    points = c["samples"]
                    diameter = max(math.dist(a["pose"][:3], b["pose"][:3]) for a in points for b in points)
                    angle = max(orientation_distance(a["pose"][3:], b["pose"][3:]) for a in points for b in points)
                    if diameter <= rules["stationary_position_m"] and angle <= rules["stationary_angle_rad"]:
                        self.event(
                            c,
                            "robot_stationary",
                            frame,
                            position_range_m=diameter,
                            angle_range_rad=angle,
                            first_frame=points[0]["frame"],
                        )
        trajectory = child["trajectory"][:steps]
        if trajectory:
            c["last_command"] = trajectory[-1]["gripper_command"]
            self.tools.state["last_gripper_command"] = c["last_command"]

    def check_direction(self, c, guard, pose, frame, reason):
        direction = guard["direction"]
        axis = "xyz".index(direction[1])
        delta = (pose[axis] - c["anchor_robot"]["eef_state"][axis]) * (1 if direction[0] == "+" else -1)
        if delta > c["rules"]["direction_tolerance_m"]:
            self.event(c, reason, frame, direction=direction, displacement_m=delta)

    def hard_reasons(self, c, status):
        reasons = [e["reason"] for e in c["guard_events"]]
        if c.get("cancel_requested"):
            return ["delegation_cancelled"]
        if status["result"]["lifecycle"] == "closed" or status["result"].get("outcome") != "incomplete":
            return list(dict.fromkeys([*reasons, "episode_ended"]))
        if status["decision"]["subgoal_version"] != c["args"]["subgoal_version"]:
            reasons.append("subgoal_changed")
        if c["steps"] >= c["args"]["budget_steps"]:
            reasons.append("step_budget")
        if self.tools.review_every_chunk and c["chunks"] >= 1:
            reasons.append("per_chunk_review")
        if not finite_robot(status["latest"]["robot"]):
            reasons.append("invalid_robot_state")
        candidate = status["decision"]["candidate"]
        if candidate is None:
            reasons.append("missing_candidate")
        if reasons:
            return list(dict.fromkeys(reasons))
        if (
            candidate["expected_frame"] != status["latest"]["frame_id"]
            or candidate["subgoal_version"] != c["args"]["subgoal_version"]
        ):
            return ["stale_candidate"]
        prefix = self.prefix(c, status)
        trajectory = candidate["eef"]["trajectory"][:prefix]
        if prefix <= 0 or len(trajectory) != prefix:
            return ["invalid_candidate_state"]
        for point in trajectory:
            pose, command = point.get("target_xyz_rpy"), point.get("gripper_command")
            if (
                not isinstance(pose, list)
                or len(pose) != 6
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in [*pose, command])
            ):
                return ["invalid_candidate_state"]
        for guard in c["args"].get("guards", []):
            if guard["kind"] == "forbidden_direction":
                self.check_direction(
                    c,
                    guard,
                    status["latest"]["robot"]["eef_state"],
                    status["latest"]["frame_id"],
                    "measured_direction_limit",
                )
                for point in trajectory:
                    self.check_direction(
                        c, guard, point["target_xyz_rpy"], status["latest"]["frame_id"], "candidate_direction_limit"
                    )
            elif guard["kind"] == "gripper_change" and guard.get("when", "after_measured") == "before_command":
                commands = [p["gripper_command"] for p in trajectory]
                if c["last_command"] is not None:
                    commands.insert(0, c["last_command"])
                if len({v > 0 for v in commands}) > 1:
                    self.event(c, "candidate_gripper_transition", status["latest"]["frame_id"])
        return list(dict.fromkeys(e["reason"] for e in c["guard_events"]))
