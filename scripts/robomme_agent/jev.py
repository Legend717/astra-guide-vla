"""Configurable HTTP-only visual advisers. Answers never authorize physical actions."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
from urllib.parse import urlsplit
import urllib.request


def post(endpoint, payload, headers, timeout):
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, allow_nan=False).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    # Explicitly bypass proxy environment for independently configured local services.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


def bounded_post(endpoint, payload, headers, timeout):
    """Bound caller latency even if a server dribbles bytes past the socket timeout.

    The daemon cannot mutate episode state. Late replies are discarded, not authorizations.
    A timed-out supervision call disables delegation for the rest of that episode.
    """
    result = queue.Queue(maxsize=1)

    def invoke():
        try:
            result.put((True, post(endpoint, payload, headers, timeout)))
        except Exception as exc:
            result.put((False, exc))

    threading.Thread(target=invoke, daemon=True).start()
    try:
        ok, value = result.get(timeout=timeout)
    except queue.Empty:
        raise TimeoutError("visual adviser deadline") from None
    if not ok:
        raise value
    return value


def validate_question(q):
    if q.get("camera", "both") not in ("front", "wrist", "both"):
        raise ValueError("invalid JEV question camera")
    labels = q["labels"]
    if (
        q["purpose"] not in ("visual_fact", "stage_warning")
        or not isinstance(q["prompt"], str)
        or not 1 <= len(q["prompt"].strip()) <= 2000
        or not 2 <= len(labels) <= 16
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
        or "uncertain" not in labels
        or not set(q["review_labels"]).issubset(labels)
        or "uncertain" not in q["review_labels"]
        or not set(labels) - set(q["review_labels"])
    ):
        raise ValueError("invalid JEV question/labels/review policy")
    for key in ("min_probability", "min_margin", "max_abstain"):
        if key in q and (type(q[key]) not in (int, float) or not 0 <= q[key] <= 1):
            raise ValueError("invalid question probability policy")


def confidence_diagnostics(probabilities, abstain, question):
    """Expose the exact heuristic gate; scores are not calibrated probabilities of visual correctness."""
    if probabilities is None:
        return {"max_probability": None, "probability_margin": None, "low_confidence_reasons": []}
    ordered = sorted(probabilities, reverse=True)
    reasons = []
    if ordered[0] < question.get("min_probability", 0):
        reasons.append("probability_below_minimum")
    if ordered[0] - ordered[1] < question.get("min_margin", 0):
        reasons.append("margin_below_minimum")
    if abstain is not None and abstain >= question.get("max_abstain", 1):
        reasons.append("abstain_at_or_above_maximum")
    return {
        "max_probability": ordered[0],
        "probability_margin": ordered[0] - ordered[1],
        "low_confidence_reasons": reasons,
    }


class VisualAdviser:
    def __init__(self, path: Path):
        self.config = json.loads(path.read_text())
        c = self.config
        if c.get("protocol") not in ("jev-choice-v1", "jev27-bare-v1", "omnijev-mso1-v1", "openai-chat"):
            raise ValueError("unsupported JEV protocol")
        url = urlsplit(c["endpoint"])
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query:
            raise ValueError("JEV endpoint must be HTTP(S), without credentials or query")
        if not c.get("model") or not c.get("revision"):
            raise ValueError("record JEV model and revision (including adapter/backend)")
        policy = c.get("probability_policy", {})
        if set(policy) - {"min_probability", "min_margin", "max_abstain"} or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in policy.values()
        ):
            raise ValueError("invalid probability policy")
        reporting = c.get("reporting_policy", {})
        interval = reporting.get("ordinary_observation_interval_chunks", 1) if isinstance(reporting, dict) else None
        if (
            not isinstance(reporting, dict)
            or set(reporting) - {"ordinary_observation_interval_chunks"}
            or type(interval) is not int
            or not 1 <= interval <= 6
        ):
            raise ValueError("invalid JEV reporting policy")
        threshold = c.get("empty_grasp_threshold_m")
        if threshold is not None and (type(threshold) not in (int, float) or not 0 < threshold <= 0.01):
            raise ValueError("invalid validated gripper threshold")
        for name in ("timeout_seconds", "max_calls", "max_total_seconds"):
            value = c[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid JEV budget: {name}")
        if type(c["max_calls"]) is not int:
            raise ValueError("max_calls must be an integer")
        if not c.get("questions"):
            raise ValueError("configure at least one JEV question")
        for q in c["questions"].values():
            if q.get("camera", "both") not in ("front", "wrist", "both"):
                raise ValueError("invalid JEV question camera")
            labels = q["labels"]
            if (
                q["purpose"] not in ("visual_fact", "stage_warning")
                or not isinstance(q["prompt"], str)
                or not q["prompt"].strip()
                or not 2 <= len(labels) <= 16
                or len(set(labels)) != len(labels)
                or any(not isinstance(label, str) or not label for label in labels)
                or "uncertain" not in labels
                or not set(q["review_labels"]).issubset(labels)
                or "uncertain" not in q["review_labels"]
            ):
                raise ValueError("invalid JEV question/labels/review policy")
        self.fingerprint = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()

    def describe(self):
        return {
            "enabled": True,
            "config_sha256": self.fingerprint,
            "model": self.config["model"],
            "revision": self.config["revision"],
            "protocol": self.config["protocol"],
            "questions": self.config["questions"],
            "reporting_policy": self.config.get("reporting_policy", {"ordinary_observation_interval_chunks": 1}),
            "mode": "advisory_only",
        }

    def observe(
        self, question: str, images: list[Path], accounting: dict, *, on_charge=lambda: None, specification=None
    ) -> dict:
        c = self.config
        q = {
            **c.get("probability_policy", {}),
            **(specification if specification is not None else c["questions"][question]),
        }
        if specification is not None:
            validate_question(q)
        base = {"question_id": question, "purpose": q["purpose"], **self.describe()}
        base.pop("questions")
        base["question_spec"] = q
        base["question_sha256"] = hashlib.sha256(json.dumps(q, sort_keys=True).encode()).hexdigest()
        if accounting["calls"] >= c["max_calls"] or accounting["seconds"] >= c["max_total_seconds"]:
            return {**base, "status": "budget_exhausted", "review_required": True}
        # Charge before issuing HTTP; timeout is still a consumed request. No automatic retry.
        accounting["calls"] += 1
        timeout = min(c["timeout_seconds"], c["max_total_seconds"] - accounting["seconds"])
        # Reserve time durably too: a process crash cannot reset the consumed service budget.
        accounting["seconds"] += timeout
        on_charge()
        start = time.perf_counter()
        try:
            encoded = []
            for path in images:
                mime = "image/png" if path.suffix == ".png" else "image/jpeg"
                encoded.append(f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode())
            if c["protocol"] in ("jev-choice-v1", "jev27-bare-v1", "omnijev-mso1-v1"):
                # This native protocol uses ONE explicitly constructed panel, not implicit multiview packing.
                if len(encoded) != 1:
                    raise ValueError("jev-choice-v1 requires one image panel")
                payload = {
                    "kind": "choice",
                    "state": [{"image": encoded[0]}],
                    "question": q["prompt"],
                    "options": q["labels"],
                    "thinking": "off",
                }
            else:
                payload = {
                    "model": c["model"],
                    "temperature": 0,
                    "max_tokens": 32,
                    "chat_template_kwargs": c.get("chat_template_kwargs", {"enable_thinking": False}),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": q["prompt"] + "\nReturn exactly one label: " + ", ".join(q["labels"]),
                                },
                                *[{"type": "image_url", "image_url": {"url": data}} for data in encoded],
                            ],
                        }
                    ],
                }
            token = os.environ.get(c.get("api_key_env", ""))
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            data = bounded_post(c["endpoint"], payload, headers, timeout)
            if c.get("require_server_identity") and (
                data.get("model") != c["model"] or data.get("revision") != c["revision"]
            ):
                raise ValueError("service identity mismatch")
            abstain = None
            if c["protocol"] in ("jev-choice-v1", "jev27-bare-v1", "omnijev-mso1-v1"):
                if c["protocol"] == "jev27-bare-v1" and data.get("protocol") != "jev27-bare-v1":
                    raise ValueError("native 27B protocol mismatch")
                probabilities = data["probabilities"]
                index = data["choice_index"]
                if c["protocol"] == "omnijev-mso1-v1":
                    abstain = data["abstain_probability"]
                    if type(abstain) not in (int, float) or not math.isfinite(abstain) or not 0 <= abstain <= 1:
                        raise ValueError("invalid abstain probability")
                if (
                    data["options"] != q["labels"]
                    or len(probabilities) != len(q["labels"])
                    or any(
                        type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities
                    )
                    or abs(sum(probabilities) + (abstain or 0) - 1) > 0.002
                    or type(index) is not int
                    or not 0 <= index < len(probabilities)
                    or probabilities[index] != max(probabilities)
                ):
                    raise ValueError("invalid native choice response")
                label = q["labels"][index]
            else:
                label = data["choices"][0]["message"]["content"].strip()
                probabilities = None  # Generated text is not a calibrated decision probability.
            if label not in q["labels"]:
                raise ValueError("invalid label")
            diagnostic = confidence_diagnostics(probabilities, abstain, q)
            low_confidence = bool(diagnostic["low_confidence_reasons"])
            label_review = label in q["review_labels"]
            result = {
                **base,
                "status": "answered",
                "label": label,
                "probabilities": probabilities,
                "abstain_probability": abstain,
                "low_confidence": low_confidence,
                **diagnostic,
                "label_review_required": label_review,
                "review_required": label_review or low_confidence,
                "server_model": data.get("model"),
                "server_revision": data.get("revision"),
            }
        except Exception as exc:
            # Do not archive HTTP bodies, credentials or server-side reasoning traces.
            result = {**base, "status": "unavailable", "error_type": type(exc).__name__, "review_required": True}
        elapsed = time.perf_counter() - start
        accounting["seconds"] += elapsed - timeout
        return {**result, "elapsed_seconds": elapsed, "accounting": dict(accounting)}
