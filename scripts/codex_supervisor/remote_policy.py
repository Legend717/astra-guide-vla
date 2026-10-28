"""Bounded client for an independently managed, session-isolated symbolic policy server."""

from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
from websockets.sync.client import connect


class RemotePolicy:
    def __init__(self, host: str, port: int, *, timeout: float = 180, mode: str = "grounded", seed: int = 42) -> None:
        self.timeout = timeout
        self.socket = connect(
            f"ws://{host}:{port}",
            proxy=None,
            compression=None,
            max_size=None,
            open_timeout=timeout,
            close_timeout=5,
        )
        try:
            self.metadata = msgpack_numpy.unpackb(self.socket.recv(timeout=timeout))
            expected = "grounded_subgoal" if mode == "grounded" else "simple_subgoal"
            if (
                self.metadata.get("protocol") != "robomme-symbolic-v1"
                or self.metadata.get("session_isolation") != "per-connection-rng"
                or self.metadata.get("actual_model_type") != expected
                or self.metadata.get("seed") != seed
            ):
                raise ValueError("external policy capabilities, symbolic mode or seed mismatch")
            self.metadata = {**self.metadata, "endpoint": f"ws://{host}:{port}", "backend": "external"}
        except BaseException:
            self.socket.close()
            raise

    def call(self, operation: str, obs: dict[str, Any]) -> dict[str, Any]:
        # A timeout poisons this connection; never resend an ambiguous inference or reconnect/reseed.
        try:
            self.socket.send(msgpack_numpy.packb({"operation": operation, "observation": obs}))
            response = self.socket.recv(timeout=self.timeout)
            if isinstance(response, str):
                raise RuntimeError("external policy returned an unexpected text response")
            result = msgpack_numpy.unpackb(response)
        except BaseException:
            self.socket.close()
            raise
        if not result.get("ok"):
            raise ValueError(
                result.get("token_budget_error", "external policy rejected request: " + result["error_type"])
            )
        return result

    def validate(self, obs: dict[str, Any]) -> None:
        self.call("validate", obs)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        result = self.call("infer", obs)
        result["actions"] = np.asarray(result["actions"])
        return result

    def close(self) -> None:
        self.socket.close()
