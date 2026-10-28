"""Persisted tool loop, bounded API usage, and deterministic episode cleanup."""

from contextlib import redirect_stdout
import copy
import fcntl
import hashlib
import io
import json
from pathlib import Path
import time
from types import SimpleNamespace

from scripts import codex_rgb_eval as rgb
from scripts import codex_symbolic_eval as bridge
from scripts.codex_supervisor.instructions import TEMPLATES
from scripts.robomme_agent import metrics
from scripts.robomme_agent.client import image_content
from scripts.robomme_agent.client import request
from scripts.robomme_agent.client import retryable
from scripts.robomme_agent.prompt import STEP_BUDGET_SYSTEM
from scripts.robomme_agent.prompt import SYSTEM
from scripts.robomme_agent.prompt import hierarchical_prompt
from scripts.robomme_agent.tools import TOOLS
from scripts.robomme_agent.tools import EpisodeTools

SOURCE_HASHES = {
    path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob("*.py"))
}


def user_message(text):
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def image_count(value):
    if isinstance(value, list):
        return sum(image_count(x) for x in value)
    if isinstance(value, dict):
        if value.get("type") == "input_image":
            return 1
        return sum(image_count(x) for x in value.values())
    return 0


def request_image_deliveries(messages):
    """Only acknowledge deliveries actually attached to this API request, never orphaned offers."""
    identifiers = set()
    for message in messages:
        content = message.get("content", message.get("output", []))
        if not isinstance(content, list) or not image_count(content):
            continue
        for item in content:
            if item.get("type") != "input_text":
                continue
            try:
                value = json.loads(item["text"])
            except (ValueError, TypeError):
                continue
            if isinstance(value, dict) and isinstance(value.get("image_delivery_id"), str):
                identifiers.add(value["image_delivery_id"])
    return sorted(identifiers)


def error_category(exc):
    """Classify known request-limit errors without persisting server bodies or credentials."""
    body = json.dumps(getattr(exc, "body", None), default=str).lower()
    if getattr(exc, "status_code", None) == 402 or any(
        marker in body
        for marker in (
            "insufficient_quota",
            "insufficient balance",
            "insufficient_balance",
            "credit balance is too low",
            "quota expired",
            "quota_expired",
            "余额不足",
            "额度已过期",
        )
    ):
        return "quota_exhausted"
    if "invalid_encrypted_content" in body:
        return "invalid_encrypted_content"
    if any(s in body for s in ("too many images", "image limit", "maximum number of images")):
        return "image_limit"
    if any(s in body for s in ("context_length", "context length", "context window", "too many tokens")):
        return "context_limit"
    return "unclassified"


class Agent:
    def __init__(self, directory, client, args):
        self.folder = directory / "agent"
        self.folder.mkdir(exist_ok=True)
        self.lock = (self.folder / "driver.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.tools = EpisodeTools(
            directory,
            timeout=args.tool_timeout,
            jev_config=args.jev_config,
            supervision_mode=getattr(args, "supervision_mode", "astra-only"),
            max_delegated_chunks=getattr(args, "max_delegated_chunks", 10),
            max_scheduled_steps=getattr(args, "max_scheduled_steps", 200),
            review_every_chunk=getattr(args, "review_every_chunk", False),
        )
        self.client = client
        self.args = args
        self.auto_evidence = getattr(args, "auto_evidence", False)
        self.instructions = hierarchical_prompt(self.tools.supervision_mode) if self.auto_evidence else SYSTEM
        if self.tools.supervision_mode == "step-budget":
            self.instructions = STEP_BUDGET_SYSTEM
            if self.tools.review_every_chunk:
                self.instructions += (
                    "\nExperimental review policy: every schedule_vla authorization returns after at most one "
                    "executed VLA chunk (up to 16 controls), even when budget_steps is larger. "
                    "The remaining budget does not execute automatically. Review the fresh evidence and issue "
                    "a new authorization or release to continue. Selected guards and other return conditions "
                    "still apply. All other tools and task objectives are unchanged.\n"
                )
        advice_allowed = self.tools.supervision_mode not in ("rules-only", "step-budget") and not (
            getattr(args, "frozen_manifest", None) and self.tools.supervision_mode == "astra-only"
        )
        self.tools.state["advice_allowed"] = advice_allowed
        self.available_tools = [
            t
            for t in TOOLS
            if (t["name"] != "supervise_vla" or self.tools.supervision_mode not in ("astra-only", "step-budget"))
            and (t["name"] not in ("schedule_vla", "history_sample") or self.tools.supervision_mode == "step-budget")
            and (t["name"] != "jev_observe" or advice_allowed)
        ]
        if self.tools.supervision_mode == "step-budget":
            self.available_tools = copy.deepcopy(self.available_tools)
            schema = next(t for t in self.available_tools if t["name"] == "experience_commit")
            properties = schema["parameters"]["properties"]["entries"]["items"]["properties"]
            properties.pop("supervision")
            properties["applicability"]["enum"] = ["common", "step-budget"]
        if getattr(args, "resume_review", None):
            if self.tools.status()["result"]["lifecycle"] != "closed":
                raise ValueError("review recovery requires a closed episode")
            if self.tools.state["pending"] or self.tools.state.get("active_delegation"):
                raise ValueError("review recovery requires settled physical requests")
            allowed = {
                "status",
                "history",
                "history_sample",
                "view_image",
                "read_memory",
                "write_memory",
                "write_review",
                "experience_commit",
                "complete",
            }
            self.available_tools = [t for t in self.available_tools if t["name"] in allowed]
        self.folder = self.tools.agent_dir
        self.path = self.folder / "conversation.json"
        self.resumed = self.path.exists()
        self.tools.state["frozen_manifest"] = metrics.validate_frozen(self.tools, args)
        self.tools.save()
        if self.resumed:
            self.state = rgb.read_json(self.path)
            if self.state["model"] != args.model:
                raise ValueError("resume requires the original model")
            if getattr(args, "resume_review", None):
                if self.state.get("pending_response"):
                    raise ValueError("resolve pending response before review recovery")
                previous = self.folder / f"report-before-review-{self.state['calls']:04d}.json"
                if not previous.exists() and (self.folder / "report.json").exists():
                    rgb.write_json(previous, rgb.read_json(self.folder / "report.json"))
                rgb.write_json(self.folder / f"review-recovery-sources-{self.state['calls']:04d}.json", SOURCE_HASHES)
                self.state["stop_reason"] = None
                errors = self.folder / "api-errors.jsonl"
                if errors.exists() and json.loads(errors.read_text().splitlines()[-1])["category"] == "image_limit":
                    self.recover_image_limit()
        else:
            context = {
                "task": self.tools.manifest["task"],
                "status": self.tools.status(),
                "templates": TEMPLATES[self.tools.manifest["task"]]["grounded"],
                "experience_update": self.tools.manifest["experience_update"],
                "experience_update_policy": self.tools.manifest.get("experience_update_policy", "always"),
                "notes": self.tools.dispatch("read_memory", {}, "init")[0],
            }
            self.state = {
                "model": args.model,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "context_tokens": None,
                "messages": [user_message(json.dumps(context))],
                "pending_response": None,
                "stop_reason": None,
            }
        rgb.write_json(
            self.folder / "config.json",
            {
                "model": args.model,
                "reasoning": args.reasoning,
                "api": "responses",
                "max_calls": args.max_calls,
                "max_output_tokens": args.max_output_tokens,
                "smoke": args.smoke,
                "max_context_tokens": args.max_context_tokens,
                "trust_env": args.trust_env,
                "image_detail": "original",
                "supervision_mode": self.tools.supervision_mode,
                "max_delegated_chunks": self.tools.max_delegated_chunks,
                "max_scheduled_steps": self.tools.max_scheduled_steps,
                "review_every_chunk": self.tools.review_every_chunk,
                "auto_evidence": self.auto_evidence,
                "experience_update_policy": self.tools.manifest.get("experience_update_policy", "always"),
                "jev_identity": {
                    key: self.tools.jev.describe()[key] for key in ("model", "revision", "protocol", "config_sha256")
                }
                if self.tools.jev
                else None,
            },
        )
        if not self.resumed:
            (self.folder / "instructions.txt").write_text(self.instructions)
            rgb.write_json(
                self.folder / "source-hashes.json",
                SOURCE_HASHES,
            )
        self.save()
        # Earlier versions also set this flag for frame/image/call-count checkpoints.
        # Those triggers are retired; only the measured context threshold can require memory now.
        if (
            not self.context_checkpoint_due()
            and not self.state.get("image_limit_checkpoint")
            and self.tools.state.get("memory_checkpoint_required")
        ):
            self.tools.state["memory_checkpoint_required"] = False
            self.tools.save()

    def record_context_usage(self, data):
        usage = data.get("usage") or {}
        incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
        self.state["context_tokens"] = incoming + outgoing if type(incoming) is int and type(outgoing) is int else None

    def context_exceeded(self):
        tokens = self.state.get("context_tokens")
        return tokens is not None and tokens > self.args.max_context_tokens

    def context_checkpoint_due(self):
        tokens = self.state.get("context_tokens")
        return tokens is not None and tokens >= self.args.max_context_tokens * 0.8

    def save(self):
        rgb.write_json(self.path, self.state)

    def recover_image_limit(self):
        """Recover only after a real API rejection; preserve newly delivered images and all archived evidence."""
        messages = self.state["messages"]
        boundary = self.state.get("acknowledged_message_count")
        if boundary is None:
            boundary = max((i for i, m in enumerate(messages) if m.get("type") == "function_call"), default=0)
        if not image_count(messages[:boundary]):
            return False
        rgb.write_json(self.folder / f"images-before-limit-{self.state['calls']:04d}.json", messages)

        def omit(value):
            if isinstance(value, list):
                return [omit(v) for v in value]
            if isinstance(value, dict):
                if value.get("type") == "input_image":
                    return {
                        "type": "input_text",
                        "text": "[Previously delivered image omitted after API image-limit rejection; re-open archive to inspect it again.]",
                    }
                return {k: omit(v) for k, v in value.items()}
            return value

        self.state["messages"] = [*[omit(m) for m in messages[:boundary]], *messages[boundary:]]
        self.state["messages"].append(
            user_message(
                "The API rejected the previous request for too many images. Previously acknowledged images are now "
                "omitted; newly supplied images remain. Write_memory using existing textual evidence and previously "
                "recorded observations, preserving uncertainties; make no fresh visual claim about omitted images. "
                "The harness will compact after this checkpoint. Re-open archived images afterwards when necessary."
            )
        )
        self.state["image_limit_checkpoint"] = True
        self.tools.state["memory_checkpoint_required"] = True
        self.tools.save()
        self.save()
        return True

    def process_response(self, response):
        """Replay a persisted response; cached tool IDs prevent duplicate physical execution."""
        function_calls = [o for o in response["output"] if o["type"] == "function_call"]
        self.state["last_successful_tools"] = []
        completed_ids = {m["call_id"] for m in self.state["messages"] if m.get("type") == "function_call_output"}
        for call in function_calls:
            if call["call_id"] in completed_ids:
                continue
            try:
                if call["name"] not in {tool["name"] for tool in self.available_tools}:
                    raise ValueError("tool is unavailable in this supervision mode")
                if len(function_calls) > 1:
                    raise ValueError("issue only ONE tool call per response; read its output before another decision")
                arguments = json.loads(call["arguments"])
                result, images = self.tools.execute(call["name"], arguments, call["call_id"])
            except (ValueError, TypeError, KeyError) as exc:
                # These errors contain only public tool arguments/protocol messages.
                result, images = {"ok": False, "error": str(exc)}, []
            if result.get("ok", True):
                self.state["last_successful_tools"].append(call["name"])
            self.state["messages"].append(
                {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": [{"type": "input_text", "text": json.dumps(result)}, *[image_content(p) for p in images]],
                }
            )
            self.save()
            print(
                json.dumps(
                    {
                        "api_call": self.state["calls"],
                        "tool": call["name"],
                        "images": len(images),
                        "ok": result.get("ok", True),
                        "frame": self.tools.status()["latest"]["frame_id"],
                    }
                ),
                flush=True,
            )
        if not function_calls:
            self.state["messages"].append(
                user_message(
                    "Continue using the RoboMME tools. If done, finish, write_review and complete. "
                    "Narrative alone does not advance or close the episode."
                )
            )
        self.state["pending_response"] = None
        self.save()

    def compact(self, *, allow_unseen=False):
        if self.state["pending_response"] or self.tools.state["pending"] or self.tools.state["image_deliveries"]:
            raise ValueError("cannot compact unresolved requests or undelivered images")
        if not allow_unseen:
            self.tools.check_seen()
        # Keep only evidence summaries written by this same agent; no second model/perception process.
        # The full transcript and every image/tool output remain in the archive.
        rgb.write_json(self.folder / f"context-before-{self.state['calls']:04d}.json", self.state["messages"])
        context = {
            "notice": "Context compacted. Same physical episode; never reset. Your full persisted memory follows. "
            "Use history/view_image to recheck uncertain details. Seen means supplied to a previous API call.",
            "memory": self.tools.dispatch("read_memory", {}, "compact")[0],
            "experience_update": self.tools.manifest["experience_update"],
            "experience_update_policy": self.tools.manifest.get("experience_update_policy", "always"),
            "status": self.tools.status(),
            "templates": TEMPLATES[self.tools.manifest["task"]]["grounded"],
            "recent_tool_results": [
                json.loads(line) for line in (self.folder / "tools.jsonl").read_text().splitlines()[-6:]
            ],
        }
        self.state["messages"] = [user_message(json.dumps(context))]
        self.state["context_tokens"] = None  # The previous request no longer measures this compacted context.
        self.state.pop("acknowledged_message_count", None)
        self.state.pop("image_limit_checkpoint", None)
        self.state["need_current_images"] = True
        self.tools.state["memory_checkpoint_required"] = False
        self.tools.save()
        self.save()

    def call_api(self):
        for attempt in range(3):
            if self.state["calls"] >= self.args.max_calls:
                self.state["stop_reason"] = "api_call_budget"
                return None
            self.state["calls"] += 1
            self.state["request_image_delivery_ids"] = request_image_deliveries(self.state["messages"])
            self.save()
            # A crash can leave an offer whose message was never saved. Discard its pending
            # bookkeeping without acknowledging it, so it cannot block later compaction.
            self.tools.state["image_deliveries"] = {
                key: value
                for key, value in self.tools.state["image_deliveries"].items()
                if key in self.state["request_image_delivery_ids"]
            }
            self.tools.save()
            started = time.monotonic()
            phase = "review" if self.tools.status()["result"]["lifecycle"] == "closed" else "execution"
            log = {"call": self.state["calls"], "attempt": attempt, "phase": phase}
            rgb.append_json(self.folder / "api-timing.jsonl", {**log, "event": "start", "epoch": time.time()})
            try:
                response = request(
                    self.client,
                    model=self.args.model,
                    instructions=self.instructions,
                    messages=self.state["messages"],
                    tools=self.available_tools,
                    reasoning=self.args.reasoning,
                    max_output_tokens=self.args.max_output_tokens,
                )
                usage = response.model_dump(mode="json", exclude_none=True).get("usage") or {}
                rgb.append_json(
                    self.folder / "api-timing.jsonl",
                    {
                        **log,
                        "event": "end",
                        "ok": True,
                        "seconds": time.monotonic() - started,
                        "cached_input_tokens": (usage.get("input_tokens_details") or {}).get("cached_tokens", 0),
                    },
                )
                return response
            except Exception as exc:
                rgb.append_json(
                    self.folder / "api-timing.jsonl",
                    {
                        **log,
                        "event": "end",
                        "ok": False,
                        "seconds": time.monotonic() - started,
                        "error_type": type(exc).__name__,
                    },
                )
                category = error_category(exc)
                can_retry = (
                    category != "quota_exhausted"
                    and retryable(exc)
                    and attempt < 2
                    and self.state["calls"] < self.args.max_calls
                )
                if (
                    error_category(exc) == "image_limit"
                    and self.tools.supervision_mode == "step-budget"
                    and not self.state.get("image_limit_checkpoint")
                    and attempt < 2
                    and self.state["calls"] < self.args.max_calls
                ):
                    can_retry = self.recover_image_limit()
                rgb.append_json(
                    self.folder / "api-errors.jsonl",
                    {
                        "attempt": attempt + 1,
                        "call": self.state["calls"],
                        "error_type": type(exc).__name__,
                        "status_code": getattr(exc, "status_code", None),
                        "category": category,
                        "retry": can_retry,
                        "retry_delay_seconds": 5 * 3**attempt if can_retry else None,
                    },
                )
                if not can_retry:
                    raise
                time.sleep(5 * 3**attempt)
        return None

    def run(self):
        error = None
        verified = False
        try:
            if self.state["pending_response"]:
                self.record_context_usage(self.state["pending_response"])
                self.tools.acknowledge_images(self.state.get("request_image_delivery_ids", []))
                if self.context_exceeded():
                    self.state["stop_reason"] = "api_context_budget"
                    self.state["pending_response"] = None
                    self.save()
                else:
                    self.process_response(self.state["pending_response"])
            elif self.resumed:
                if "context_tokens" not in self.state and self.state.get("since_compaction", 0):
                    previous = sorted(self.folder.glob("response-*.json"))
                    if previous:
                        self.record_context_usage(rgb.read_json(previous[-1]))
                self.state["messages"].append(user_message(json.dumps({"resumed_same_episode": self.tools.status()})))
            while (
                self.state["stop_reason"] != "api_context_budget"
                and not self.tools.state["complete"]
                and self.state["calls"] < self.args.max_calls
            ):
                if self.context_exceeded():
                    self.state["stop_reason"] = "api_context_budget"
                    break
                status = self.tools.status()
                if (
                    self.args.smoke
                    and self.tools.state["reviewed_intervention"]
                    and status["result"]["lifecycle"] != "closed"
                ):
                    self.tools.finish("smoke takeover and post-intervention review completed")
                    self.state["messages"].append(
                        user_message(
                            "Engineering smoke closed after a real intervention, all returned history, updated memory "
                            "and a fresh candidate. Write_review and complete. This is not a task-success evaluation."
                        )
                    )
                if (
                    self.state["calls"] >= self.args.max_calls - 3
                    and self.tools.status()["result"]["lifecycle"] != "closed"
                ):
                    self.state["stop_reason"] = "api_call_budget"
                    self.tools.finish("API call budget: reserve final calls for review")
                    self.state["messages"].append(
                        user_message("Call budget reached. Episode closed; write_review then complete.")
                    )
                # Preserve the full prefix/cache until the measured context reaches the threshold.
                # Only then ask for fresh model-written memory before rebuilding the context.
                if self.context_checkpoint_due() or self.state.get("image_limit_checkpoint"):
                    recent = self.state.get("last_successful_tools", [])
                    if (
                        "write_memory" in recent
                        and not self.tools.state["pending"]
                        and not self.tools.state["image_deliveries"]
                        and (self.auto_evidence or not self.tools.status()["review_required_count"])
                    ):
                        self.compact(allow_unseen=self.auto_evidence)
                    else:
                        self.tools.state["memory_checkpoint_required"] = True
                        self.tools.save()
                        self.state["messages"].append(
                            user_message(
                                "Context checkpoint due: write_memory with the evidence ALREADY delivered, "
                                "task conditions, identity/order/counts, frame IDs, uncertainties and next step. "
                                "Unseen history remains pending and will follow after compaction. Do not claim to have "
                                "reviewed unseen frames. Save memory before a physical decision or further image queries."
                            )
                        )
                if self.auto_evidence and not self.tools.state["pending"]:
                    status = self.tools.status()
                    if (
                        status["review_required_count"]
                        or self.state.get("need_current_images")
                        or status["latest"]["frame_id"] not in self.tools.state["original_front"]
                    ):
                        bundle, images = self.tools.handoff(f"auto-{self.state['calls']}", max_frames=128)
                        self.state["messages"].append(
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": json.dumps(bundle)},
                                    *[image_content(p) for p in images],
                                ],
                            }
                        )
                        self.state["need_current_images"] = False
                        self.save()
                response = self.call_api()
                if response is None:
                    break
                data = response.model_dump(mode="json", exclude_none=True)
                rgb.write_json(self.folder / f"response-{self.state['calls']:04d}.json", data)
                usage = data.get("usage") or {}
                self.state["input_tokens"] += usage.get("input_tokens", 0)
                self.state["output_tokens"] += usage.get("output_tokens", 0)
                self.record_context_usage(data)
                if self.context_exceeded():
                    # Exact image/reasoning token usage is known only after the response. Archive it,
                    # but never execute tools produced with an over-budget context.
                    self.tools.acknowledge_images(self.state.get("request_image_delivery_ids", []))
                    self.state["stop_reason"] = "api_context_budget"
                    self.save()
                    break
                if data["status"] != "completed":
                    self.state["stop_reason"] = "incomplete_api_response"
                    break  # Never execute partial/truncated function arguments.
                self.state["last_tools"] = [o["name"] for o in data["output"] if o["type"] == "function_call"]
                self.state["messages"].extend(data["output"])
                self.state["pending_response"] = data
                self.save()
                self.tools.acknowledge_images(self.state.get("request_image_delivery_ids", []))
                self.state["acknowledged_message_count"] = len(self.state["messages"]) - len(data["output"])
                self.process_response(data)
        except (Exception, KeyboardInterrupt) as exc:
            error = type(exc).__name__  # API exceptions may include URLs or secrets: never archive their text.
            self.state["stop_reason"] = "agent_error"
        finally:
            if (
                not self.state["stop_reason"]
                and self.state["calls"] >= self.args.max_calls
                and not self.tools.state["complete"]
            ):
                self.state["stop_reason"] = "api_call_budget"
            try:
                self.tools.finish(self.state["stop_reason"] or "agent completed or exhausted budget")
                with redirect_stdout(io.StringIO()) as output:
                    bridge.verify(SimpleNamespace(run_dir=self.tools.directory))
                (self.folder / "verification.txt").write_text(output.getvalue())
                verified = True
            except Exception as exc:
                error = error or type(exc).__name__
            self.save()
            self.lock.close()
        status = self.tools.status()
        smoke_passed = bool(
            self.tools.state["reviewed_intervention"]
            and self.tools.state["review_written"]
            and self.tools.state["complete"]
            and verified
            and not error
            and status["result"]["outcome"] != "error"
        )
        report = {
            "model": self.args.model,
            "policy_backend": self.tools.manifest.get("policy_backend", "real"),
            "api_calls": self.state["calls"],
            "input_tokens": self.state["input_tokens"],
            "output_tokens": self.state["output_tokens"],
            "error_type": error,
            "stop_reason": self.state["stop_reason"],
            "verified": verified,
            "smoke_requested": self.args.smoke,
            "takeover_smoke_passed": smoke_passed,
            "post_intervention_review": self.tools.state["reviewed_intervention"],
            "interventions": self.tools.state["interventions"],
            "result": status["result"],
            "review_written_by_model": self.tools.state["review_written"],
            "metrics": metrics.summarize(self.tools),
        }
        rgb.write_json(self.folder / "report.json", report)
        if not self.tools.state["review_written"]:
            (self.tools.directory / "review.md").write_text(
                "# Harness termination report\n\nThe model did not finish a final review. "
                "This machine report makes no visual claims.\n\n```json\n" + json.dumps(report, indent=2) + "\n```\n"
            )
        print(json.dumps(report, indent=2), flush=True)
        complete = self.tools.state["complete"] and self.tools.state["review_written"]
        healthy = verified and not error and status["result"]["outcome"] != "error"
        return 0 if healthy and complete and (not self.args.smoke or smoke_passed) else 1
