"""External grounded pi0.5 server: one weight tree, isolated RNG per WebSocket connection."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import http
import logging
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openpi_client import msgpack_numpy  # noqa: E402
import websockets.asyncio.server as server  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402

from scripts.codex_supervisor.policy import LocalPolicy  # noqa: E402


class SymbolicServer:
    def __init__(self, policy: LocalPolicy) -> None:
        self.policy = policy
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi-inference")
        self.metadata = {
            **policy.metadata,
            "protocol": "robomme-symbolic-v1",
            "session_isolation": "per-connection-rng",
            "capabilities": ["validate", "infer", "reset"],
        }
        # This adapter deliberately supports only symbolic policies, which have no mutable history buffer.
        if policy.policy.config.representation_type != "symbolic" or policy.policy.mem_buffer is not None:
            raise ValueError("session isolation requires a symbolic policy without a history buffer")
        self.initial_rng = policy.policy._rng  # noqa: SLF001

    def dispatch(self, request: dict[str, Any], rng: Any) -> tuple[dict[str, Any], Any]:
        operation = request.get("operation")
        try:
            if operation == "reset":
                return {"ok": True, "reset_finished": True}, self.initial_rng
            observation = request["observation"]
            if operation == "validate":
                self.policy.validate(observation)
                return {"ok": True}, rng
            if operation != "infer":
                raise ValueError("unknown operation")
            # All model access runs on one executor. JAX keys are immutable and stored per connection.
            self.policy.policy._rng = rng  # noqa: SLF001
            result = self.policy.infer(observation)
            return {"ok": True, "actions": result["actions"]}, self.policy.policy._rng  # noqa: SLF001
        except Exception as exc:
            result = {"ok": False, "error_type": type(exc).__name__}
            if isinstance(exc, ValueError) and re.fullmatch(
                r"Full model text requires \d+ tokens; limit is \d+", str(exc)
            ):
                result["token_budget_error"] = str(exc)
            # Failures do not mutate the session's committed random state.
            return result, rng

    async def handler(self, websocket: server.ServerConnection) -> None:
        rng = self.initial_rng
        await websocket.send(msgpack_numpy.packb(self.metadata))
        try:
            async for message in websocket:
                request = msgpack_numpy.unpackb(message)
                result, rng = await asyncio.get_running_loop().run_in_executor(
                    self.executor, self.dispatch, request, rng
                )
                await websocket.send(msgpack_numpy.packb(result))
        except ConnectionClosed:
            pass

    async def run(self, host: str, port: int) -> None:
        def health(connection, request):
            if request.path == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        async with server.serve(
            self.handler, host, port, compression=None, max_size=None, process_request=health
        ) as service:
            logging.info("Ready on %s:%s, shared weights / per-connection RNG", host, port)
            await service.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="mme_vla_suite")
    parser.add_argument("--mode", choices=("grounded", "simple-smoke"), default="grounded")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    policy = LocalPolicy(args.checkpoint.resolve(), args.config, args.mode, args.seed)
    service = SymbolicServer(policy)
    try:
        asyncio.run(service.run(args.host, args.port))
    finally:
        service.executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
