"""Paused episode state machine. Only release/intervene call the environment."""

import time
import uuid

import numpy as np

from scripts import codex_rgb_eval as rgb
from scripts.codex_supervisor.instructions import convert


class Supervisor:
    def __init__(self, directory, recorder, runtime, policy, fk, *, task, mode, max_steps, prefix):
        self.directory = directory
        self.recorder = recorder
        self.runtime = runtime
        self.policy = policy
        self.fk = fk
        self.task = task
        self.mode = mode
        self.max_steps = max_steps
        self.prefix = prefix
        self.version = 0
        self.subgoal = None
        self.specification = None
        self.candidate = None
        self.actions = None
        self.closed = False
        self.state = {
            "ready_epoch": time.time(),
            "lifecycle": "ready",
            "outcome": "incomplete",
            "control_steps": 0,
            "inference_count": 0,
            "feedback": rgb.safe_feedback({}),
            "recorded_frames": recorder.frame_count,
            "video_finalized": False,
        }
        self.publish()

    @property
    def frame(self):
        return self.recorder.latest["frame_id"]

    def publish(self):
        self.state.update(
            recorded_frames=self.recorder.frame_count,
            subgoal_version=self.version,
            candidate_id=None if self.candidate is None else self.candidate["candidate_id"],
        )
        rgb.write_json(self.directory / "result.json", self.state)
        rgb.write_json(
            self.directory / "decision.json",
            {
                "expected_frame": self.frame,
                "subgoal_version": self.version,
                "subgoal": self.subgoal,
                "candidate": self.candidate,
                "result": self.state,
            },
        )

    def infer(self):
        self.candidate = self.actions = None
        if self.closed or self.subgoal is None:
            return
        self.state["lifecycle"] = "inferring"
        self.state["inference_count"] += 1
        self.publish()
        started = time.monotonic()
        actions = np.asarray(self.policy.infer(self.runtime.observation(self.subgoal))["actions"], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 8 or not 1 <= len(actions) <= 256 or not np.isfinite(actions).all():
            raise ValueError("invalid post-transform VLA action chunk")
        identifier = uuid.uuid4().hex
        self.actions = actions.copy()
        description = self.fk.describe(actions, self.recorder.latest["robot"]["eef_state"])
        self.candidate = {
            "candidate_id": identifier,
            "expected_frame": self.frame,
            "subgoal_version": self.version,
            "predicted_length": len(actions),
            "default_prefix": min(self.prefix, len(actions)),
            "inference_seconds": time.monotonic() - started,
            "eef": description,
        }
        rgb.write_json(
            self.directory / "internal" / f"candidate-{identifier}.json",
            {
                **self.candidate,
                "actions": actions.tolist(),
                "subgoal": self.subgoal,
            },
        )
        rgb.append_json(self.directory / "events.jsonl", {"event": "candidate", **self.candidate})
        self.state["lifecycle"] = "ready"
        self.publish()

    def step(self, action, kind, request_id):
        index = self.state["control_steps"]
        self.state["control_steps"] += 1
        before = self.frame
        started = time.monotonic()
        rgb.append_json(
            self.directory / "internal/controls.jsonl",
            {
                "request_id": request_id,
                "control_step": index,
                "kind": kind,
                "action": list(action),
                "before_frame": before,
            },
        )
        obs, _reward, terminated, truncated, info = self.runtime.step(action, kind)
        control_error = rgb.recoverable_control_error(info)
        feedback = (
            {**rgb.safe_feedback({}), "control_error": control_error}
            if control_error
            else rgb.safe_feedback(info, terminated=terminated, truncated=truncated)
        )
        self.recorder.video_context = {
            "source": "VLA" if kind == "joint" else "INTERVENTION",
            "request_id": request_id,
            "subgoal": self.subgoal["grounded_subgoal"],
        }
        self.recorder.capture(obs, info, f"step_{index:07d}", feedback)
        self.state.update(feedback=feedback, outcome=rgb.outcome(feedback))
        if rgb.outcome(feedback) != "incomplete":
            self.closed = True
        if self.state["control_steps"] >= self.max_steps:
            self.closed = True
            self.state["stop_reason"] = "control_budget"
        if self.closed:
            self.state.setdefault("physical_ended_epoch", time.time())
        rgb.append_json(
            self.directory / "events.jsonl",
            {
                "event": "control_return",
                "request_id": request_id,
                "kind": kind,
                "control_step": index,
                "before_frame": before,
                "after_frame": self.frame,
                "feedback": feedback,
                "seconds": time.monotonic() - started,
            },
        )
        return control_error is not None or self.closed

    def handle(self, request, request_id):
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        operation = request.get("operation")
        if operation == "finish":
            self.closed = True
            self.state.setdefault("physical_ended_epoch", time.time())
            self.candidate = self.actions = None
            self.state["stop_reason"] = "user_finish"
            self.publish()
            return {"ok": True}
        if self.closed:
            raise ValueError("episode has ended")
        if type(request.get("expected_frame")) is not int or request["expected_frame"] != self.frame:
            raise ValueError("stale expected_frame")
        if type(request.get("subgoal_version")) is not int or request["subgoal_version"] != self.version:
            raise ValueError("stale subgoal_version")
        if operation not in ("prompt", "release", "intervene"):
            raise ValueError("choose release, prompt or intervene")
        if request.get("candidate_id") != (self.candidate or {}).get("candidate_id"):
            raise ValueError("stale candidate_id")
        started = time.monotonic()
        if operation == "prompt":
            specification = request.get("subgoal", {})
            updated = convert(self.task, specification, self.runtime.image_hw())
            if self.mode == "simple-smoke" and updated["simple_subgoal"] is None:
                raise ValueError("simple-smoke requires an explicit paired simple_subgoal")
            self.policy.validate(self.runtime.observation(updated))
            previous = self.specification
            self.candidate = self.actions = None
            self.subgoal = updated
            self.specification = specification
            self.version += 1
            rgb.append_json(
                self.directory / "events.jsonl",
                {
                    "event": "prompt",
                    "request_id": request_id,
                    "frame": self.frame,
                    "subgoal_version": self.version,
                    "specification": specification,
                    "converted": updated,
                    "semantic_change": previous is None or previous.get("text") != specification.get("text"),
                    "coordinate_change": previous is None
                    or previous.get("point_yx_1000") != specification.get("point_yx_1000"),
                },
            )
            self.infer()
            return {"ok": True, "advanced_steps": 0}
        if self.candidate is None:
            raise ValueError("set a prompt to generate a candidate before deciding")
        predicted = len(self.actions)
        if operation == "release":
            length = request.get("prefix", min(self.prefix, predicted))
            if type(length) is not int or not 1 <= length <= min(self.prefix, predicted):
                raise ValueError("prefix must be a nonempty prefix no longer than the default execution length")
            commands = [("joint", row.tolist()) for row in self.actions[:length]]
        else:
            sequence = request.get("sequence")
            if not isinstance(sequence, list) or not 1 <= len(sequence) <= 16:
                raise ValueError("intervene requires 1 to 16 bounded end-effector actions")
            if any(not isinstance(item, dict) for item in sequence):
                raise ValueError("each intervention item must be an object")
            commands = [rgb.validate_action(item) for item in sequence]
            length = sum(item[1] for item in commands)
            if length > 100:
                raise ValueError("intervention sequence may contain at most 100 total steps")
        candidate_id = self.candidate["candidate_id"]
        self.candidate = self.actions = None  # Invalidate BEFORE any step; never reuse a tail.
        self.state["lifecycle"] = "executing"
        self.publish()
        rgb.append_json(
            self.directory / "events.jsonl",
            {
                "event": operation,
                "request_id": request_id,
                "candidate_id": candidate_id,
                "subgoal_version": self.version,
                "frame": self.frame,
                "predicted_length": predicted,
                "authorized_length": length,
                "reason": request.get("reason", ""),
            },
        )
        executed = 0
        interrupted = False
        for command in commands:
            if operation == "release":
                targets = [command[1]]
            else:
                target, steps, interpolate = command
                start = self.recorder.latest["robot"]["eef_state"]
                targets = [rgb.action_at(start, target, i, interpolate) for i in range(steps)]
            for action in targets:
                if self.closed:
                    interrupted = True
                    break
                interrupted = self.step(action, "joint" if operation == "release" else "eef", request_id)
                executed += 1
                if interrupted:
                    break
            if interrupted:
                break
        rgb.append_json(
            self.directory / "events.jsonl",
            {
                "event": "execution_end",
                "request_id": request_id,
                "candidate_id": candidate_id,
                "predicted_length": predicted,
                "authorized_length": length,
                "actual_length": executed,
                "discarded_tail": predicted - executed if operation == "release" else predicted,
                "after_frame": self.frame,
                "seconds": time.monotonic() - started,
            },
        )
        self.infer()
        self.publish()
        return {"ok": True, "executed_steps": executed}
