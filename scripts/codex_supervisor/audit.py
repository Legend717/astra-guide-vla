"""Offline audit of state transitions and original-action prefix execution."""

import hashlib
import json

from scripts import codex_rgb_eval as rgb


def verify_events(directory, events, result):
    reset = rgb.read_json(directory / "observations/reset/batch.json")
    frame = reset["first_frame"] + reset["frame_count"] - 1
    version = 0
    candidate = None
    archives = {}
    active = None
    returned = 0
    controls_path = directory / "internal/controls.jsonl"
    controls = [json.loads(line) for line in controls_path.read_text().splitlines()] if controls_path.exists() else []
    if len(controls) != result["control_steps"]:
        raise ValueError("raw control count mismatch")
    for event in events:
        kind = event["event"]
        if kind == "request_started":
            path = directory / "requests" / f"{event['request_id']}.json"
            if hashlib.sha256(path.read_bytes()).hexdigest() != event["sha256"]:
                raise ValueError("request body changed after processing")
        elif kind == "prompt":
            if active or event["frame"] != frame or event["subgoal_version"] != version + 1:
                raise ValueError("prompt advanced state or has an invalid version")
            version += 1
            candidate = None
        elif kind == "candidate":
            identifier = event["candidate_id"]
            if (
                active
                or event["expected_frame"] != frame
                or event["subgoal_version"] != version
                or identifier in archives
            ):
                raise ValueError("candidate state/version/identity mismatch")
            archive = rgb.read_json(directory / "internal" / f"candidate-{identifier}.json")
            if (
                archive["candidate_id"] != identifier
                or archive["expected_frame"] != frame
                or archive["subgoal_version"] != version
            ):
                raise ValueError("candidate archive identity mismatch")
            if not len(archive["actions"]) == event["predicted_length"] == len(archive["eef"]["trajectory"]):
                raise ValueError("candidate archive length mismatch")
            archives[identifier] = archive
            candidate = identifier
        elif kind in ("release", "intervene"):
            if (
                active
                or event["candidate_id"] != candidate
                or event["frame"] != frame
                or event["subgoal_version"] != version
            ):
                raise ValueError("executed an invalidated candidate")
            active = {**event, "returned": 0}
            if event["predicted_length"] != len(archives[candidate]["actions"]):
                raise ValueError("predicted length mismatch")
            candidate = None
        elif kind == "control_return":
            if active is None or event["request_id"] != active["request_id"] or event["control_step"] != returned:
                raise ValueError("control return has no matching authorization")
            if event["before_frame"] != frame or not frame <= event["after_frame"] < result["recorded_frames"]:
                raise ValueError("control observation frames are not continuous")
            raw = controls[returned]
            expected_kind = "joint" if active["event"] == "release" else "eef"
            if (
                raw["request_id"] != active["request_id"]
                or raw["kind"] != expected_kind
                or raw["control_step"] != returned
                or raw["before_frame"] != frame
            ):
                raise ValueError("raw control and public return disagree")
            if expected_kind == "joint":
                original = archives[active["candidate_id"]]["actions"][active["returned"]]
                if raw["action"] != original:
                    raise ValueError("released action differs from the original candidate prefix")
            active["returned"] += 1
            returned += 1
            frame = event["after_frame"]
        elif kind == "execution_end":
            if active is None or event["request_id"] != active["request_id"]:
                raise ValueError("execution end has no authorization")
            if (
                event["candidate_id"] != active["candidate_id"]
                or event["actual_length"] != active["returned"]
                or not 0 <= event["actual_length"] <= active["authorized_length"]
                or event["authorized_length"] != active["authorized_length"]
                or event["after_frame"] != frame
            ):
                raise ValueError("execution lengths or final frame mismatch")
            tail = active["predicted_length"] - (active["returned"] if active["event"] == "release" else 0)
            if event["discarded_tail"] != tail:
                raise ValueError("discarded tail mismatch")
            active = None
    if active or returned != result["control_steps"] or len(archives) != result["inference_count"]:
        raise ValueError("incomplete execution or inference archive")
    if frame != result["recorded_frames"] - 1:
        raise ValueError("final archived frame mismatch")
