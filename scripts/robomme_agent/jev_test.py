"""HTTP contract tests; no model service, GPU or network required."""

import json
from pathlib import Path

import pytest

from scripts.robomme_agent import jev
from scripts.robomme_agent.jev import VisualAdviser


@pytest.mark.parametrize("camera", [None, "history", "left", 0, True])
def test_question_rejects_invalid_camera(camera):
    q = {
        "prompt": "Is the cube held?",
        "purpose": "visual_fact",
        "camera": camera,
        "labels": ["held", "empty", "uncertain"],
        "review_labels": ["held", "uncertain"],
    }
    with pytest.raises(ValueError, match="camera"):
        jev.validate_question(q)


@pytest.fixture
def adviser(tmp_path):
    config = json.loads((Path(__file__).parent / "profiles/jev27-local.json").read_text())
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(config))
    return VisualAdviser(path)


def respond(monkeypatch, body, seen):
    def handler(endpoint, payload, headers, timeout):
        seen.append(payload)
        return body

    monkeypatch.setattr(jev, "post", handler)


def test_native_validation_and_budget(adviser, tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    image.write_bytes(b"mock image")
    labels = adviser.config["questions"]["contact_readiness"]["labels"]
    seen = []
    respond(monkeypatch, {"options": labels, "probabilities": [0.8, 0.1, 0.1], "choice_index": 0}, seen)
    accounting = {"calls": 0, "seconds": 0.0}
    charges = []
    result = adviser.observe(
        "contact_readiness", [image], accounting, on_charge=lambda: charges.append(accounting["calls"])
    )
    assert result["review_required"]
    assert result["label"] == "approaching_contact"
    assert charges == [1]
    assert seen[0]["thinking"] == "off"
    accounting["calls"] = adviser.config["max_calls"]
    assert adviser.observe("contact_readiness", [image], accounting)["status"] == "budget_exhausted"
    assert len(seen) == 1


@pytest.mark.parametrize(
    ("probabilities", "abstain", "reasons"),
    [
        ([0.6, 0.3, 0.1], 0.0, []),
        ([0.5771, 0.348, 0.0734], 0.0015, ["probability_below_minimum"]),
        ([0.41, 0.4, 0.19], 0.0, ["probability_below_minimum", "margin_below_minimum"]),
        ([0.65, 0.1, 0.05], 0.2, ["abstain_at_or_above_maximum"]),
    ],
)
def test_confidence_diagnostics_explains_actual_gate(probabilities, abstain, reasons):
    result = jev.confidence_diagnostics(
        probabilities, abstain, {"min_probability": 0.6, "min_margin": 0.1, "max_abstain": 0.2}
    )
    assert result["low_confidence_reasons"] == reasons
    assert result["max_probability"] == max(probabilities)


def test_chat_does_not_invent_confidence_scores():
    assert jev.confidence_diagnostics(None, None, {}) == {
        "max_probability": None,
        "probability_margin": None,
        "low_confidence_reasons": [],
    }


@pytest.mark.parametrize(
    "reporting",
    [
        None,
        [],
        {"ordinary_observation_interval_chunks": 0},
        {"ordinary_observation_interval_chunks": True},
        {"ordinary_observation_interval_chunks": 7},
        {"ignore_uncertain": True},
    ],
)
def test_invalid_observation_cadence_rejected(adviser, tmp_path, reporting):
    config = {**adviser.config, "reporting_policy": reporting}
    path = tmp_path / "bad-cadence.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="reporting policy"):
        VisualAdviser(path)


@pytest.mark.parametrize(("probabilities", "index"), [([float("nan"), 0, 1], 2), ([0.1, 0.8, 0.1], 0), ([1, 0], 0)])
def test_invalid_is_review(adviser, tmp_path, monkeypatch, probabilities, index):
    image = tmp_path / "image.png"
    image.write_bytes(b"mock image")
    labels = adviser.config["questions"]["contact_readiness"]["labels"]
    respond(monkeypatch, {"options": labels, "probabilities": probabilities, "choice_index": index}, [])
    result = adviser.observe("contact_readiness", [image], {"calls": 0, "seconds": 0.0})
    assert result["status"] == "unavailable"
    assert result["review_required"]


def test_chat_uses_configured_model_without_fabricated_probability(adviser, tmp_path, monkeypatch):
    adviser.config.update(protocol="openai-chat", model="ordinary-backbone")
    image = tmp_path / "image.png"
    image.write_bytes(b"mock image")
    seen = []
    respond(monkeypatch, {"choices": [{"message": {"content": "clearly_separated"}}]}, seen)
    result = adviser.observe("contact_readiness", [image], {"calls": 0, "seconds": 0.0})
    assert result["probabilities"] is None
    assert not result["review_required"]
    assert seen[0]["model"] == "ordinary-backbone"


def test_timeout_charged_once(adviser, tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    image.write_bytes(b"mock")
    calls = []

    def fail(*args):
        calls.append(args)
        raise TimeoutError("do not archive this body")

    monkeypatch.setattr(jev, "post", fail)
    accounting = {"calls": 0, "seconds": 0.0}
    result = adviser.observe("contact_readiness", [image], accounting)
    assert result["review_required"]
    assert result["error_type"] == "TimeoutError"
    assert accounting["calls"] == len(calls) == 1
    assert "do not archive" not in json.dumps(result)


def test_http_transport_uses_explicit_endpoint_and_timeout(monkeypatch):
    import io

    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append((request, timeout))
            return io.BytesIO(b'{"ok": true}')

    monkeypatch.setattr(jev.urllib.request, "build_opener", lambda *_: Opener())
    assert jev.post("http://localhost:123/v1/decide", {"kind": "choice"}, {}, 2) == {"ok": True}
    request, timeout = requests[0]
    assert request.full_url == "http://localhost:123/v1/decide"
    assert timeout == 2
    assert json.loads(request.data) == {"kind": "choice"}


@pytest.mark.parametrize("native_protocol", ["jev27-bare-v1", "wrong-native-protocol"])
def test_native27_protocol_and_choice_only_probabilities(adviser, tmp_path, monkeypatch, native_protocol):
    adviser.config.update(protocol="jev27-bare-v1", require_server_identity=True)
    image = tmp_path / "image.png"
    image.write_bytes(b"mock")
    labels = adviser.config["questions"]["contact_readiness"]["labels"]
    response = {
        "model": adviser.config["model"],
        "revision": adviser.config["revision"],
        "protocol": native_protocol,
        "options": labels,
        "probabilities": [0.05, 0.9, 0.05],
        "choice_index": 1,
    }
    respond(monkeypatch, response, [])
    result = adviser.observe("contact_readiness", [image], {"calls": 0, "seconds": 0.0})
    if native_protocol == "jev27-bare-v1":
        assert result["status"] == "answered"
        assert not result["review_required"]
        assert result["abstain_probability"] is None
    else:
        assert result["status"] == "unavailable"
        assert result["review_required"]


@pytest.mark.parametrize("question_override", [None, 0.4])
def test_presets_inherit_reporting_policy_with_question_override(adviser, tmp_path, monkeypatch, question_override):
    adviser.config["probability_policy"] = {"min_probability": 0.6, "min_margin": 0.1}
    question = adviser.config["questions"]["contact_readiness"]
    if question_override is not None:
        question["min_probability"] = question_override
    image = tmp_path / "image.png"
    image.write_bytes(b"mock")
    respond(monkeypatch, {"options": question["labels"], "probabilities": [0.2, 0.45, 0.35], "choice_index": 1}, [])
    result = adviser.observe("contact_readiness", [image], {"calls": 0, "seconds": 0.0})
    assert result["status"] == "answered"
    assert result["label"] == "clearly_separated"
    assert result["review_required"] is (question_override is None)
