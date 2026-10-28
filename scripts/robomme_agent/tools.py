"""Allowlisted model tools. Never expose shell, arbitrary files, or simulator internals."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time

from scripts import codex_rgb_eval as rgb
from scripts import codex_symbolic_eval as bridge
from scripts.codex_supervisor import experience
from scripts.codex_supervisor.history import render_history
from scripts.codex_supervisor.history import render_sampled_history
from scripts.codex_supervisor.instructions import annotate
from scripts.codex_supervisor.instructions import convert
from scripts.robomme_agent.scheduling import PROPERTIES as SCHEDULE_PROPERTIES
from scripts.robomme_agent.scheduling import RULES
from scripts.robomme_agent.scheduling import SCHEDULING_EXPERIENCE
from scripts.robomme_agent.scheduling import StepScheduler
from scripts.robomme_agent.supervision import CONTRACT_PROPERTIES
from scripts.robomme_agent.supervision import QUESTION
from scripts.robomme_agent.supervision import Delegator


def function(name, description, properties=None, required=None):
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": properties or {},
            "required": required if required is not None else list(properties or {}),
            "additionalProperties": False,
        },
    }


INTEGER = {"type": "integer", "minimum": 0}
TEXT = {"type": "string"}
IDENTITY = {
    "expected_frame": INTEGER,
    "subgoal_version": INTEGER,
    "candidate_id": {"type": ["string", "null"]},
}
SUBGOAL = {
    "type": "object",
    "properties": {
        "text": TEXT,
        "point_yx_1000": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "number", "minimum": 0, "maximum": 1000},
                "minItems": 2,
                "maxItems": 2,
            },
        },
    },
    "required": ["text", "point_yx_1000"],
    "additionalProperties": False,
}
TOOLS = [
    function(
        "schedule_vla",
        "Authorize 30..200 actual VLA controls in multiples of ten (configured cap in status). "
        "Each fresh candidate executes at most 16 controls, capped by remaining budget. "
        "Optional guards return early at chunk boundaries; no guard proves success. "
        "gripper_change.when: before_command or after_measured (default); forbidden_direction.direction: "
        "+x/-x/+y/-y/+z/-z in WORLD coordinates relative to this decision. stationary: opt-in 32-control window.",
        {**IDENTITY, **SCHEDULE_PROPERTIES},
        required=[*IDENTITY, "budget_steps", "reason"],
    ),
    function(
        "history_sample",
        "View a sparse chronological front/wrist history, including interval endpoints. "
        "Start with max_frames=8, then narrow the interval to find event frames; read originals for detail. "
        "Unsampled frames are NOT reviewed. Does not advance simulation.",
        {"first": INTEGER, "last": INTEGER, "max_frames": {"type": "integer", "minimum": 2, "maximum": 32}},
        required=["first", "last"],
    ),
    function(
        "supervise_vla",
        "Delegate bounded VLA execution to JEV (rules-only in the experiment baseline). "
        "Normal default: 6 chunks; hard cap from status, normally 10. Every chunk uses a fresh candidate. "
        "Questions should describe local visible facts or early stage warnings, with uncertain requiring review. "
        "max_chunks is the only execution budget. The harness uses the candidate default prefix capped at 16 "
        "steps (4 in timing mode); JEV never changes the prefix. Critical/recovery/timing: max_chunks=1. "
        "Use release for close Astra supervision or post-intervention recovery. No automatic subgoal changes.",
        {**IDENTITY, **CONTRACT_PROPERTIES},
        required=[*IDENTITY, *(key for key in CONTRACT_PROPERTIES if key != "questions")],
    ),
    function(
        "jev_observe",
        "Ask a configured advisory visual-fact/stage question on the CURRENT camera image. "
        "Does not replace your image review or authorize action. See status for question IDs.",
        {"expected_frame": INTEGER, "question": TEXT, "camera": {"type": "string", "enum": ["front", "wrist"]}},
    ),
    function("status", "Read current official task, robot state, filtered result, pending FK and unseen frame ranges."),
    function(
        "history",
        "View ALL front/wrist frames in an inclusive range, up to 64 at once, as image contact sheets.",
        {"first": INTEGER, "last": INTEGER},
    ),
    function(
        "view_image",
        "View one ORIGINAL executed camera frame, at full original size.",
        {"frame": INTEGER, "camera": {"type": "string", "enum": ["front", "wrist"]}},
    ),
    function(
        "inspect_subgoal",
        "Mark only your supplied points on the current original front image, without executing.",
        {"subgoal": SUBGOAL},
    ),
    function(
        "set_subgoal",
        "Validate grounded subgoal, invalidate old candidate and infer anew WITHOUT moving.",
        {**IDENTITY, "subgoal": SUBGOAL, "reason": TEXT},
    ),
    function(
        "release",
        "Execute the reviewed current VLA candidate prefix, then infer a NEW candidate.",
        {**IDENTITY, "prefix": {"type": "integer", "minimum": 1}, "reason": TEXT},
    ),
    function(
        "intervene",
        "Execute bounded absolute world xyz/rpy/gripper actions, then infer anew and resume supervision.",
        {
            **IDENTITY,
            "reason": TEXT,
            "sequence": {
                "type": "array",
                "minItems": 1,
                "maxItems": 16,
                "items": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "array", "items": {"type": "number"}, "minItems": 7, "maxItems": 7},
                        "steps": {"type": "integer", "minimum": 1, "maximum": 100},
                        "interpolate": INTEGER,
                    },
                    "required": ["target", "steps", "interpolate"],
                    "additionalProperties": False,
                },
            },
        },
    ),
    function(
        "query",
        "Query the original request ID. Pending means do not submit another physical action.",
        {"request_id": TEXT},
    ),
    function("read_memory", "Read the episode memory, review and allowed experience snapshot."),
    function(
        "write_memory",
        "Replace episode notes with evidence, hypotheses, phase and next step. Cite the reviewed current candidate ID, or null if none.",
        {
            "text": TEXT,
            "frames": {"type": "array", "items": INTEGER, "minItems": 1},
            "candidate_id": {"type": ["string", "null"]},
        },
    ),
    function(
        "write_review",
        "Write the final review after closing the episode, citing observations and limitations.",
        {"text": TEXT, "frames": {"type": "array", "items": INTEGER, "minItems": 1}},
    ),
    function(
        "finish", "Close the physical episode and finalize its video. Does not declare task success.", {"reason": TEXT}
    ),
    function(
        "experience_commit",
        "After closing, append evidenced experience when updates are enabled. Mock: engineering only.",
        {
            "scope": {"type": "string", "enum": ["global", "task"]},
            "entries": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": TEXT,
                        "conditions": TEXT,
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                        "kind": {"type": "string", "enum": ["observation", "hypothesis", "engineering"]},
                        "frames": {"type": "array", "items": INTEGER, "minItems": 1},
                        "phase": TEXT,
                        "applicability": {"type": "string", "enum": ["common", "jev", "step-budget"]},
                        "scheduling": SCHEDULING_EXPERIENCE,
                        "supervision": {
                            "type": "object",
                            "properties": {
                                "recommended_chunks": {"type": "integer", "minimum": 1},
                                "prefix_steps": {"type": "integer", "minimum": 1, "maximum": 16},
                                "questions": {"type": "array", "items": QUESTION},
                                "astra_watch": {"type": "boolean"},
                                "evidence_count": {"type": "integer", "minimum": 1},
                            },
                            "additionalProperties": False,
                        },
                        "supersedes": {"type": "array", "items": TEXT},
                        "status": {"type": "string", "enum": ["active", "contradicted", "disabled"]},
                    },
                    "required": ["claim", "conditions", "confidence", "kind", "frames"],
                    "additionalProperties": False,
                },
            },
        },
        required=["entries"],
    ),
    function("complete", "End the agent after the closed episode has a final review."),
]


class EpisodeTools:
    def __init__(
        self,
        directory: Path,
        *,
        timeout: float = 180,
        jev_config: Path | None = None,
        supervision_mode: str = "astra-only",
        max_delegated_chunks: int = 10,
        max_scheduled_steps: int = 200,
        review_every_chunk: bool = False,
    ):
        self.directory = directory
        self.agent_dir = directory / "agent"
        self.agent_dir.mkdir(exist_ok=True)
        self.timeout = timeout
        self.supervision_mode = supervision_mode
        self.max_delegated_chunks = max_delegated_chunks
        self.max_scheduled_steps = max_scheduled_steps
        self.review_every_chunk = review_every_chunk
        if review_every_chunk and supervision_mode != "step-budget":
            raise ValueError("per-chunk review requires step-budget mode")
        self.supervision_config = {}
        self.manifest = rgb.read_json(directory / "manifest.json")
        self.state_path = self.agent_dir / "tools-state.json"
        self.state = (
            rgb.read_json(self.state_path)
            if self.state_path.exists()
            else {
                "seen": [],
                "original_front": [],
                "pending": None,
                "interventions": [],
                "reviewed_intervention": False,
                "review_written": False,
                "complete": False,
            }
        )
        self.state.setdefault("image_deliveries", {})
        self.state.setdefault("original_wrist", [])
        self.jev = None
        if jev_config:
            if supervision_mode == "step-budget":
                raise ValueError("step-budget mode does not accept JEV configuration; select jev explicitly")
            from scripts.robomme_agent.jev import VisualAdviser

            self.jev = VisualAdviser(jev_config)
        if supervision_mode == "jev" and self.jev is None:
            raise ValueError("JEV delegation requires --jev-config")
        supervision_identity = {"mode": supervision_mode, "max_chunks": max_delegated_chunks}
        if supervision_mode == "step-budget":
            supervision_identity = {"mode": supervision_mode, "max_steps": max_scheduled_steps, "rules": RULES}
            if review_every_chunk:
                supervision_identity["review_every_chunk"] = True
        if self.state.get("supervision_config", supervision_identity) != supervision_identity:
            raise ValueError("resume requires the original supervision mode and limits")
        self.state["supervision_config"] = supervision_identity
        fingerprint = self.jev.fingerprint if self.jev else None
        if "jev_config_sha256" in self.state and self.state["jev_config_sha256"] != fingerprint:
            raise ValueError("resume requires the original JEV configuration; compare models in separate runs")
        self.state["jev_config_sha256"] = fingerprint
        self.state.setdefault("jev_accounting", {"calls": 0, "seconds": 0.0})
        if self.jev:
            # API keys are environment references only; never serialize their values.
            rgb.write_json(self.agent_dir / "jev-config.json", self.jev.config)

    def delegator(self):
        return StepScheduler(self) if self.supervision_mode == "step-budget" else Delegator(self)

    def experience_entries(self):
        entries = []
        for key, filename in (
            ("experience_input", "experience-input.json"),
            ("task_experience_input", "experience-task-input.json"),
        ):
            if self.manifest.get(key, {}).get("enabled"):
                entries.extend(
                    experience.active_view(rgb.read_json(self.directory / filename), mode=self.supervision_mode)[
                        "entries"
                    ]
                )
        return entries

    def acknowledge_images(self, identifiers=None):
        """Called only after the API returns: a filename/tool offer is not image delivery."""
        # The API driver passes the immutable delivery IDs from that request, including on replay.
        # None is retained for explicit acknowledgments by local callers/tests.
        identifiers = list(self.state["image_deliveries"]) if identifiers is None else identifiers
        for identifier in identifiers:
            delivery = self.state["image_deliveries"].pop(identifier, None)
            if delivery is None:
                continue
            self.state["seen"] = sorted(set(self.state["seen"]) | set(delivery["frames"]))
            self.state["original_front"] = sorted(set(self.state["original_front"]) | set(delivery["front"]))
            self.state["original_wrist"] = sorted(set(self.state["original_wrist"]) | set(delivery.get("wrist", [])))
            # Separately delivered originals establish a reviewed pair only after BOTH actually reached Astra.
            if self.supervision_mode == "step-budget":
                self.state["seen"] = sorted(
                    set(self.state["seen"]) | (set(self.state["original_front"]) & set(self.state["original_wrist"]))
                )
        self.save()

    def save(self):
        rgb.write_json(self.state_path, self.state)

    def status(self):
        latest = rgb.read_json(self.directory / "latest.json")
        decision = rgb.read_json(self.directory / "decision.json")
        unseen = sorted(set(range(latest["frame_id"] + 1)) - set(self.state["seen"]))
        return {
            "latest": latest,
            "decision": decision,
            "result": rgb.read_json(self.directory / "result.json"),
            "pending": self.state["pending"],
            "policy": self.manifest.get("policy"),
            "experience_updates": experience.update_policy(
                self.manifest, rgb.read_json(self.directory / "result.json")
            ),
            "jev": {**self.jev.describe(), "advice_allowed": self.state.get("advice_allowed", True)}
            if self.jev
            else {"enabled": False},
            "supervision": {
                **self.state["supervision_config"],
                "jev_disabled": self.state.get("jev_disabled", False),
                "active_contract": self.state.get("active_delegation"),
            },
            "unseen_count": len(unseen),
            "review_required_count": int(latest["frame_id"] not in self.state["seen"])
            if self.supervision_mode == "step-budget"
            else len(unseen),
            "history_available": {"first": 0, "last": latest["frame_id"]},
            "next_history": {"first": unseen[0], "last": min(unseen[0] + 63, unseen[-1])} if unseen else None,
        }

    def frame(self, index):
        if type(index) is not int or index < 0:
            raise ValueError("frame must be a nonnegative integer")
        for line in (self.directory / "frames.jsonl").read_text().splitlines():
            item = json.loads(line)
            if item["frame_id"] == index:
                return item
        raise ValueError("frame does not exist")

    def image(self, relative):
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory.resolve()) or path.suffix not in (".jpg", ".png"):
            raise ValueError("invalid archived image")
        return path

    def check_seen(self):
        if self.status()["review_required_count"]:
            if self.supervision_mode == "step-budget":
                raise ValueError("view the current front/wrist pair before deciding; older history is optional")
            raise ValueError("view ALL outstanding history before deciding or claiming post-action review")

    def handoff(self, identifier, *, max_frames=128):
        """Supply a bounded chronological history batch and current originals; the API loop acknowledges delivery."""
        status = self.status()
        latest = status["latest"]
        if self.supervision_mode == "step-budget":
            frame = latest["frame_id"]
            self.state["image_deliveries"][identifier] = {"frames": [frame], "front": [frame], "wrist": [frame]}
            self.save()
            return {
                "image_delivery_id": identifier,
                "status": status,
                "delivered_history": [frame],
                "current_originals": frame,
                "history_policy": "Current views only. Use history_sample to review process or reset demonstration.",
            }, [self.image(latest["rgb"][camera]) for camera in ("front", "wrist")]
        unseen = sorted(set(range(latest["frame_id"] + 1)) - set(self.state["seen"]))[:max_frames]
        if unseen:
            # A manual history query can create gaps. Do not render a huge range across those gaps.
            for index, frame in enumerate(unseen):
                if frame != unseen[0] + index:
                    unseen = unseen[:index]
                    break
        images = []
        for start in range(0, len(unseen), 64):
            batch = unseen[start : start + 64]
            rendered = render_history(self.directory, batch[0], batch[-1])
            images.extend(self.image(p) for p in rendered["contact_sheets"])
        images.extend(self.image(latest["rgb"][camera]) for camera in ("front", "wrist"))
        self.state["image_deliveries"][identifier] = {
            "frames": unseen,
            "front": [latest["frame_id"]],
            "wrist": [latest["frame_id"]],
        }
        self.save()
        return {
            "image_delivery_id": identifier,
            "status": status,
            "delivered_history": unseen,
            "current_originals": latest["frame_id"],
        }, images

    def wait(self, request_id):
        path = self.directory / "responses" / f"{request_id}.json"
        deadline = time.monotonic() + self.timeout
        while not path.exists():
            if time.monotonic() > deadline:
                return {"pending": True, "request_id": request_id}
            if rgb.read_json(self.directory / "result.json")["lifecycle"] == "closed":
                break
            time.sleep(0.1)
        if not path.exists():
            return {"pending": True, "request_id": request_id, "server_closed": True}
        result = rgb.read_json(path)
        self.state["pending"] = None
        # A stable ID also makes recovery after interrupted waiting idempotent.
        req = rgb.read_json(self.directory / "requests" / f"{request_id}.json")
        if (
            req["operation"] == "intervene"
            and result.get("ok")
            and not any(x["request_id"] == request_id for x in self.state["interventions"])
        ):
            latest = rgb.read_json(self.directory / "latest.json")
            before = self.frame(req["expected_frame"])["robot"]
            changed = any(
                abs(a - b) > 1e-5
                for key in ("eef_state", "joint_state", "gripper_state")
                for a, b in zip(before[key], latest["robot"][key], strict=True)
            )
            self.state["interventions"].append(
                {
                    "request_id": request_id,
                    "before": req["expected_frame"],
                    "after": latest["frame_id"],
                    "candidate_before": req["candidate_id"],
                    "executed_steps": result.get("executed_steps", 0),
                    "robot_state_changed": changed,
                    "control_error": result.get("result", {}).get("feedback", {}).get("control_error"),
                }
            )
        if self.supervision_mode == "step-budget" and result.get("ok") and result.get("executed_steps", 0):
            controls = self.directory / "internal/controls.jsonl"
            if controls.exists():
                for line in reversed(controls.read_text().splitlines()):
                    control = json.loads(line)
                    if control.get("request_id") == request_id:
                        self.state["last_gripper_command"] = control["action"][-1]
                        break
        self.save()
        return {**result, "status": self.status()}

    def submit(self, operation, args, request_id):
        if self.state["pending"] and self.state["pending"] != request_id:
            raise ValueError("query the original pending request before another submission")
        request = {**args, "operation": operation, "client_id": self.manifest["client_id"]}
        existing = self.directory / "requests" / f"{request_id}.json"
        if existing.exists() and rgb.read_json(existing) != request:
            raise ValueError("request ID already exists with a different body")
        # Persist intent before enqueueing. Reattaching can safely recreate the SAME immutable request.
        self.state["pending"] = request_id
        self.save()
        bridge.immutable_request(self.directory, request_id, request)
        return self.wait(request_id)

    def finish(self, reason):
        active = self.state.get("active_delegation")
        if active:
            contract = self.state["delegations"][active]
            contract["cancel_requested"] = True
            self.save()
            settled, _ = self.delegator().run(contract["args"], active)
            if settled.get("pending"):
                return settled
        if self.state["pending"]:
            pending = self.wait(self.state["pending"])
            if pending.get("pending"):
                return pending
        result = rgb.read_json(self.directory / "result.json")
        if result["lifecycle"] != "closed":
            existing = self.directory / "requests/agent-finish.json"
            if existing.exists():
                reason = rgb.read_json(existing).get("reason", reason)
            result = self.submit("finish", {"reason": reason}, "agent-finish")
            deadline = time.monotonic() + self.timeout
            while rgb.read_json(self.directory / "result.json")["lifecycle"] != "closed":
                if time.monotonic() > deadline:
                    return {"pending": True, "request_id": "agent-finish"}
                time.sleep(0.1)
        return self.status()

    def execute(self, name, args, call_id):
        if not isinstance(args, dict):
            raise ValueError("tool arguments must be an object")
        schema = next((t for t in TOOLS if t["name"] == name), None)
        if schema is None:
            raise ValueError("unknown tool")
        params = schema["parameters"]
        if set(args) - set(params["properties"]) or set(params["required"]) - set(args):
            raise ValueError("missing or unsupported tool arguments")
        validate_schema(args, params)
        identifier = "api-" + hashlib.sha256(call_id.encode()).hexdigest()[:32]
        cache = self.agent_dir / f"{identifier}.json"
        if cache.exists():
            saved = rgb.read_json(cache)
            if saved["name"] != name or saved["args"] != args:
                raise ValueError("tool call ID reused with a different body")
            result = saved["result"]
            if saved["images"]:
                result = {**result, "image_delivery_id": identifier}
                if saved.get("image_delivery"):
                    self.state["image_deliveries"].setdefault(identifier, saved["image_delivery"])
                    self.save()
            return result, [self.image(p) for p in saved["images"]]
        result, images = self.dispatch(name, args, identifier)
        if images:
            result = {**result, "image_delivery_id": identifier}
        self.save()
        saved = {
            "name": name,
            "args": args,
            "result": result,
            "images": [str(p.relative_to(self.directory)) for p in images],
            "image_delivery": self.state["image_deliveries"].get(identifier),
        }
        rgb.write_json(cache, saved)
        rgb.append_json(self.agent_dir / "tools.jsonl", {"call_id": call_id, **saved})
        return result, images

    def dispatch(self, name, args, identifier):
        if name == "schedule_vla":
            return StepScheduler(self).run(args, identifier)
        if name == "supervise_vla":
            if self.supervision_mode == "step-budget":
                raise ValueError("legacy delegation is disabled; use schedule_vla")
            return Delegator(self).run(args, identifier)
        if name == "jev_observe":
            if not self.state.get("advice_allowed", True):
                raise ValueError("JEV advice is disabled in this experiment baseline")
            if self.jev is None:
                raise ValueError("JEV disabled; launch with --jev-config")
            if self.state["pending"]:
                raise ValueError("resolve pending physical request before JEV observation")
            latest = rgb.read_json(self.directory / "latest.json")
            if args["expected_frame"] != latest["frame_id"]:
                raise ValueError("stale JEV observation request")
            if args["question"] not in self.jev.config["questions"]:
                raise ValueError("unknown JEV question")
            source = self.image(latest["rgb"][args["camera"]])
            result = self.jev.observe(args["question"], [source], self.state["jev_accounting"], on_charge=self.save)
            result.update(
                {
                    "frame_id": latest["frame_id"],
                    "camera": args["camera"],
                    "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                }
            )
            rgb.append_json(self.agent_dir / "jev-observations.jsonl", result)
            if result["status"] == "answered":
                self.state["jev_delivered_frames"] = sorted(
                    set(self.state.get("jev_delivered_frames", [])) | {latest["frame_id"]}
                )
            # JEV delivery is deliberately NOT an Astra image-delivery acknowledgment.
            return result, []
        if name == "status":
            return self.status(), []
        if name == "history_sample":
            result = render_sampled_history(self.directory, args["first"], args["last"], args.get("max_frames", 8))
            self.state["image_deliveries"][identifier] = {"frames": result["selected_frames"], "front": []}
            return result, [self.image(p) for p in result["contact_sheets"]]
        if name == "history":
            first, last = args["first"], args["last"]
            if type(first) is not int or type(last) is not int or not 0 <= first <= last < first + 64:
                raise ValueError("history requires 1..64 contiguous frames")
            result = render_history(self.directory, first, last)
            self.state["image_deliveries"][identifier] = {"frames": list(range(first, last + 1)), "front": []}
            return result, [self.image(p) for p in result["contact_sheets"]]
        if name == "view_image":
            if args["camera"] not in ("front", "wrist"):
                raise ValueError("camera must be front or wrist")
            frame = self.frame(args["frame"])
            self.state["image_deliveries"][identifier] = {
                "frames": [],
                "front": [args["frame"]] if args["camera"] == "front" else [],
                "wrist": [args["frame"]] if args["camera"] == "wrist" else [],
            }
            return frame, [self.image(frame["rgb"][args["camera"]])]
        if name == "inspect_subgoal":
            from PIL import Image

            latest = rgb.read_json(self.directory / "latest.json")
            source = self.image(latest["rgb"]["front"])
            with Image.open(source) as im:
                converted = convert(self.manifest["task"], args["subgoal"], (im.height, im.width))
            path = self.agent_dir / f"{identifier}.png"
            annotate(source, path, converted["pixel_yx"])
            return {"frame": latest["frame_id"], **converted}, [path]
        if name in ("set_subgoal", "release", "intervene"):
            if self.state.get("active_delegation"):
                raise ValueError("query the pending delegation before a new decision")
            operation = "prompt" if name == "set_subgoal" else name
            # On recovery, don't recheck stale frame/history against an ALREADY submitted control.
            if (self.directory / "requests" / f"{identifier}.json").exists():
                return self.submit(operation, args, identifier), []
            if self.state.get("memory_checkpoint_required"):
                raise ValueError("save reviewed evidence with write_memory before the next physical decision")
            self.check_seen()
            if name == "set_subgoal" and args["expected_frame"] not in self.state["original_front"]:
                raise ValueError("open the current ORIGINAL front image before selecting subgoal coordinates")
            return self.submit(operation, args, identifier), []
        if name == "query":
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args["request_id"]):
                raise ValueError("invalid request ID")
            active = self.state.get("active_delegation")
            if active:
                c = self.state["delegations"][active]
                if not c["next_request"] or c["next_request"]["id"] != args["request_id"]:
                    raise ValueError("query the original delegation child request")
                return self.delegator().run(c["args"], active)
            if self.state["pending"] != args["request_id"]:
                path = self.directory / "responses" / f"{args['request_id']}.json"
                return (rgb.read_json(path) if path.exists() else {"pending": True}), []
            return self.wait(args["request_id"]), []
        if name == "read_memory":
            return {
                "memory": (self.directory / "memory.md").read_text(),
                "review": (self.directory / "review.md").read_text(),
                "experience": experience.active_view(
                    rgb.read_json(self.directory / "experience-input.json"), mode=self.supervision_mode
                )
                if self.manifest["experience_input"]["enabled"]
                else None,
                "task_experience": experience.active_view(
                    rgb.read_json(self.directory / "experience-task-input.json"), mode=self.supervision_mode
                )
                if self.manifest.get("task_experience_input", {}).get("enabled")
                else None,
            }, []
        if name in ("write_memory", "write_review"):
            if name == "write_review":
                self.check_seen()
            frames = args["frames"]
            delivered = set(self.state["seen"]) | set(self.state["original_front"]) | set(self.state["original_wrist"])
            if (
                not isinstance(frames, list)
                or not frames
                or any(type(f) is not int or f not in delivered for f in frames)
            ):
                raise ValueError("notes must cite seen frames")
            if not isinstance(args["text"], str) or not 1 <= len(args["text"]) <= 40000:
                raise ValueError("notes must contain 1..40000 characters")
            if name == "write_review" and self.status()["result"]["lifecycle"] != "closed":
                raise ValueError("close the episode before final review")
            path = self.directory / ("memory.md" if name == "write_memory" else "review.md")
            path.write_text(args["text"] + "\n")
            if name == "write_review":
                self.state["review_written"] = True
            else:
                self.state["memory_checkpoint_required"] = False
            if name == "write_memory" and self.state["interventions"]:
                intervention = self.state["interventions"][-1]
                current = self.status()
                candidate = current["decision"]["candidate"]
                if (
                    not current["review_required_count"]
                    and intervention["after"] > intervention["before"]
                    and intervention["after"] in frames
                    and candidate
                    and candidate["candidate_id"] != intervention["candidate_before"]
                    and args["candidate_id"] == candidate["candidate_id"]
                    and intervention["robot_state_changed"]
                    and intervention["executed_steps"] > 0
                    and not intervention["control_error"]
                ):
                    self.state["reviewed_intervention"] = True
                    self.state["reviewed_candidate"] = candidate["candidate_id"]
            return {"saved": path.name, "frames": frames}, []
        if name == "finish":
            return self.finish(args["reason"]), []
        if name == "experience_commit":
            if self.supervision_mode == "step-budget":
                args = {**args, "entries": [dict(entry) for entry in args["entries"]]}
                for entry in args["entries"]:
                    if entry.get("applicability") == "jev" or "supervision" in entry:
                        raise ValueError("step-budget runs cannot commit legacy JEV scheduling")
                    entry.setdefault("applicability", "step-budget" if "scheduling" in entry else "common")
            delivered = set(self.state["seen"]) | set(self.state["original_front"]) | set(self.state["original_wrist"])
            if any(f not in delivered for entry in args["entries"] for f in entry["frames"]):
                raise ValueError("experience must cite frames delivered to the model")
            return experience.commit(self.directory, args), []
        if name == "complete":
            if not self.state["review_written"] or self.status()["result"]["lifecycle"] != "closed":
                raise ValueError("finish and write_review before complete")
            self.state["complete"] = True
            return {"complete": True}, []
        raise ValueError("unknown tool")


def validate_schema(value, schema):
    """Validate the small JSON-schema subset used above, including nested action arguments."""
    types = schema.get("type")
    types = types if isinstance(types, list) else [types]
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": type(value) is int,
        "number": type(value) in (int, float) and math.isfinite(value),
        "null": value is None,
        "boolean": type(value) is bool,
    }
    if not any(valid.get(t, False) for t in types):
        raise ValueError("tool argument has an invalid type")
    if "multipleOf" in schema and value % schema["multipleOf"]:
        raise ValueError("tool number must be a multiple of " + str(schema["multipleOf"]))
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("tool argument is outside the allowed choices")
    if isinstance(value, dict):
        properties = schema["properties"]
        if set(schema.get("required", [])) - set(value) or set(value) - set(properties):
            raise ValueError("missing or unsupported nested tool fields")
        for key, item in value.items():
            validate_schema(item, properties[key])
    elif isinstance(value, list):
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf")):
            raise ValueError("tool array has an invalid length")
        for item in value:
            validate_schema(item, schema["items"])
    elif type(value) in (int, float):
        if not math.isfinite(value) or not schema.get("minimum", -math.inf) <= value <= schema.get("maximum", math.inf):
            raise ValueError("tool number is outside the allowed range")
