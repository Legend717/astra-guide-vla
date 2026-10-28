"""External policy session isolation and client lifecycle contracts."""

from types import SimpleNamespace
from typing import ClassVar

import numpy as np
from openpi_client import msgpack_numpy
import pytest

from scripts import run_robomme_agent as cli
from scripts.codex_supervisor.remote_policy import RemotePolicy
from scripts.serve_robomme_policy import SymbolicServer


@pytest.fixture
def service():
    class Policy:
        metadata: ClassVar[dict] = {"actual_model_type": "grounded_subgoal", "seed": 42}
        policy = SimpleNamespace(config=SimpleNamespace(representation_type="symbolic"), mem_buffer=None, _rng=42)

        def validate(self, observation):
            if observation.get("too_long"):
                raise ValueError("Full model text requires 201 tokens; limit is 200")

        def infer(self, observation):
            self.validate(observation)
            value = self.policy._rng  # noqa: SLF001
            self.policy._rng += 1  # noqa: SLF001
            return {"actions": np.full((32, 8), value)}

    service = SymbolicServer(Policy())
    yield service
    service.executor.shutdown()


def test_interleaved_sessions_reset_and_validate_do_not_cross_rng(service):
    a = b = service.initial_rng
    first_a, a = service.dispatch({"operation": "infer", "observation": {}}, a)
    first_b, b = service.dispatch({"operation": "infer", "observation": {}}, b)
    np.testing.assert_array_equal(first_a["actions"], first_b["actions"])
    _, b = service.dispatch({"operation": "reset"}, b)
    _, a = service.dispatch({"operation": "validate", "observation": {}}, a)
    next_a, a = service.dispatch({"operation": "infer", "observation": {}}, a)
    reset_b, b = service.dispatch({"operation": "infer", "observation": {}}, b)
    assert next_a["actions"][0, 0] == 43
    assert reset_b["actions"][0, 0] == 42
    rejected, a_after = service.dispatch({"operation": "validate", "observation": {"too_long": True}}, a)
    assert rejected["token_budget_error"] == "Full model text requires 201 tokens; limit is 200"
    assert a_after == a


def test_remote_close_only_disconnects_and_timeout_never_retries(monkeypatch, service):
    class Socket:
        def __init__(self):
            self.sent = []
            self.closed = 0

        def recv(self, *, timeout):
            if not self.sent:
                return msgpack_numpy.packb(service.metadata)
            raise TimeoutError

        def send(self, message):
            self.sent.append(msgpack_numpy.unpackb(message))

        def close(self):
            self.closed += 1

    socket = Socket()
    monkeypatch.setattr("scripts.codex_supervisor.remote_policy.connect", lambda *args, **kwargs: socket)
    client = RemotePolicy("127.0.0.1", 8000, timeout=0.01)
    with pytest.raises(TimeoutError):
        client.infer({})
    assert len(socket.sent) == 1
    assert socket.sent[0]["operation"] == "infer"
    assert socket.closed == 1
    client.close()
    assert len(socket.sent) == 1  # No stop/reset message and no server lifecycle action.


def test_incompatible_server_rejected_and_connection_closed(monkeypatch):
    socket = SimpleNamespace(recv=lambda **kwargs: msgpack_numpy.packb({}), close=lambda: closed.append(True))
    closed = []
    monkeypatch.setattr("scripts.codex_supervisor.remote_policy.connect", lambda *args, **kwargs: socket)
    with pytest.raises(ValueError, match="capabilities"):
        RemotePolicy("127.0.0.1", 8000)
    assert closed == [True]


def test_harness_launch_passes_port_without_starting_or_loading_policy(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.bridge, "OUTPUT_ROOT", tmp_path)
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        directory = tmp_path / "episode" / "artifacts"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text('{"lifecycle":"ready"}')
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(cli.subprocess, "Popen", launch)
    args = cli.parser().parse_args(
        ["--policy-backend", "real", "--policy-port", "8123", "--run-dir", str(tmp_path / "episode")]
    )
    args.max_steps = 600
    directory, process = cli.launch(args)
    assert process.pid == 123
    assert directory == tmp_path / "episode/artifacts"
    assert len(commands) == 1
    assert "--policy-port" in commands[0]
    assert "8123" in commands[0]
    assert "--checkpoint" not in commands[0]
    assert "--policy-python" not in commands[0]
    assert not any("serve_robomme_policy" in argument for argument in commands[0])


def test_default_output_layout_uses_official_seed_and_custom_root(monkeypatch, tmp_path):
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        directory = cli.Path(command[command.index("--run-dir") + 1]) / "artifacts"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text('{"lifecycle":"ready"}')
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(cli.subprocess, "Popen", launch)
    args = cli.parser().parse_args(["--output-root", str(tmp_path), "--episode", "0"])
    args.max_steps = 1
    first, _ = cli.launch(args)
    second, _ = cli.launch(args)
    assert first.parent.parent == tmp_path / "PickHighlight"
    assert first.parent.name.startswith("ep0_seed620000_")
    assert second != first
    assert commands[0][commands[0].index("--output-root") + 1] == str(tmp_path)
    assert cli.parser().parse_args([]).output_root == cli.ROOT / "runs/evaluation/harness"


def test_official_seed_uses_selected_episode_and_split():
    assert cli.episode_seed("PickHighlight", "test", 0) == 620000
    assert cli.episode_seed("PickHighlight", "test", 1) == 620100
    assert cli.episode_seed("PickHighlight", "val", 0) != 620000
    with pytest.raises(ValueError, match="official seed"):
        cli.episode_seed("PickHighlight", "test", 999999)
