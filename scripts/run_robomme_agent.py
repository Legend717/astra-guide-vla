"""Run the standalone API supervisor and one paused RoboMME simulator episode."""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import codex_rgb_eval as rgb  # noqa: E402
from scripts import codex_symbolic_eval as bridge  # noqa: E402
from scripts.robomme_agent.client import create_client  # noqa: E402
from scripts.robomme_agent.runner import Agent  # noqa: E402

HARNESS_ROOT = ROOT / "runs/evaluation/harness"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=rgb.TASKS, default="PickHighlight")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--dataset", choices=("test", "val", "train"), default="test")
    p.add_argument("--output-root", type=Path, default=HARNESS_ROOT, help="Root for generated task/episode directories")
    p.add_argument("--run-dir", type=Path, help="Exact new output directory; overrides --output-root")
    p.add_argument("--attach", type=Path, help="Resume API driver of an existing LIVE episode; never resets it")
    p.add_argument("--resume-review", type=Path, help="Resume only archived review of a CLOSED episode; no simulator")
    p.add_argument("--policy-backend", choices=("mock-stall", "real"), default="mock-stall")
    p.add_argument("--policy-host", default="127.0.0.1")
    p.add_argument("--policy-port", type=int, help="External pi service port; required for real backend")
    p.add_argument("--policy-timeout", type=float, default=180)
    p.add_argument("--sim-python", type=Path, default=Path(os.environ.get("ROBOMME_SIM_PYTHON", sys.executable)))
    p.add_argument("--cpu-render-launcher", type=Path, default=ROOT / "scripts/run_simulator.sh")
    p.add_argument("--env-file", type=Path, default=ROOT / ".env")
    p.add_argument("--model", default="gpt-6-astra")
    p.add_argument("--reasoning", choices=("low", "medium", "high", "xhigh", "max"), default="high")
    p.add_argument(
        "--trust-env", action="store_true", help="Honor HTTP proxy environment (SOCKS requires httpx[socks])"
    )
    p.add_argument("--api-timeout", type=float, default=180)
    p.add_argument("--tool-timeout", type=float, default=180)
    p.add_argument(
        "--jev-config", type=Path, help="Optional HTTP visual-adviser JSON profile; no model packages needed"
    )
    p.add_argument(
        "--supervision-mode", choices=("step-budget", "astra-only", "jev", "rules-only"), default="step-budget"
    )
    p.add_argument("--auto-evidence", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-delegated-chunks", type=int, default=10)
    p.add_argument("--max-scheduled-steps", type=int, default=200)
    p.add_argument(
        "--review-every-chunk", action="store_true", help="Return each step-budget authorization after one chunk"
    )
    p.add_argument("--task-experience", type=Path)
    p.add_argument("--frozen-manifest", type=Path, help="Immutable experiment identities and experience hashes")
    p.add_argument("--startup-timeout", type=float, default=300)
    p.add_argument("--max-calls", type=int, default=80)
    p.add_argument("--max-output-tokens", type=int, default=8192)
    p.add_argument(
        "--max-context-tokens",
        type=int,
        default=300000,
        help="Per-response input plus output context limit; cumulative token use is accounting only",
    )
    p.add_argument("--max-steps", type=int, help="Default: 256 for --smoke, otherwise 1300")
    p.add_argument("--prefix", type=int, default=16)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Stop after intervention, new real images and model review of a fresh candidate",
    )
    p.add_argument("--experience", type=Path)
    p.add_argument("--experience-input", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--experience-update", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--experience-update-policy",
        choices=("always", "failure-only"),
        default="always",
        help="Allow updates after any closed episode, or only unsuccessful episodes (including abandonment)",
    )
    return p


def episode_seed(task: str, dataset: str, episode: int) -> int:
    """Read only the official episode seed, without importing simulator or GPU dependencies."""
    path = ROOT / "third_party/robomme_benchmark/src/robomme/env_metadata" / dataset
    metadata = json.loads((path / f"record_dataset_{task}_metadata.json").read_text())
    records = {
        (str(record.get("task") or metadata.get("env_id") or ""), int(record["episode"])): record
        for record in metadata.get("records", [])
        if record.get("episode") is not None
    }
    record = records.get((task, episode))
    if record is None or record.get("seed") is None:
        raise ValueError("episode has no official seed in the selected split")
    return int(record["seed"])


def launch(args):
    if args.run_dir is not None:
        run = args.run_dir.resolve()
        output_root = run.parent
    else:
        seed = episode_seed(args.task, args.dataset, args.episode)
        output_root = args.output_root.resolve()
        run = (
            output_root
            / args.task
            / (f"ep{args.episode}_seed{seed}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}")
        )
    if run.exists():
        raise ValueError("run-dir must be a new directory; existing archives are never overwritten")
    experience_root = (
        ROOT / "runs/evaluation/step-budget/experience"
        if args.supervision_mode == "step-budget"
        else bridge.OUTPUT_ROOT / "experience"
    )
    experience = args.experience or experience_root / (
        "agent-engineering.json" if args.policy_backend != "real" else "general.json"
    )
    command = [
        "bash",
        str(args.cpu_render_launcher),
        str(args.sim_python),
        str(ROOT / "scripts/codex_symbolic_eval.py"),
        "serve",
        "--run-dir",
        str(run),
        "--output-root",
        str(output_root),
        "--client-id",
        "api-agent",
        "--task",
        args.task,
        "--dataset",
        args.dataset,
        "--episode",
        str(args.episode),
        "--max-steps",
        str(args.max_steps),
        "--prefix",
        str(args.prefix),
        "--policy-backend",
        args.policy_backend,
        "--policy-host",
        args.policy_host,
        "--policy-timeout",
        str(args.policy_timeout),
        "--experience",
        str(experience),
        "--experience-input" if args.experience_input else "--no-experience-input",
        "--experience-update" if args.experience_update else "--no-experience-update",
        "--experience-update-policy",
        args.experience_update_policy,
    ]
    if args.policy_port is not None:
        command.extend(["--policy-port", str(args.policy_port)])
    task_experience = args.task_experience or experience.parent / "tasks" / f"{args.task}.json"
    command.extend(["--task-experience", str(task_experience)])
    env = dict(os.environ)
    if args.policy_backend != "real":
        env["CUDA_VISIBLE_DEVICES"] = ""
    # The server creates run itself. Logs live outside it until initialization, and are never model tools.
    with tempfile.NamedTemporaryFile(prefix="robomme-agent-", suffix=".log", delete=False) as log:
        process = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, env=env
        )
    directory = run / "artifacts"
    deadline = time.monotonic() + args.startup_timeout
    try:
        while True:
            path = directory / "result.json"
            result = rgb.read_json(path) if path.exists() else {}
            if result.get("lifecycle") == "ready":
                rgb.write_json(
                    directory / "agent-host.json",
                    {
                        "pid": process.pid,
                        "log": log.name,
                        "client_id": "api-agent",
                        "command": command,
                    },
                )
                print(f"Simulator ready: {run}", flush=True)
                return directory, process
            if process.poll() is not None or result.get("lifecycle") == "closed":
                raise RuntimeError(f"simulator startup failed; trusted operator log: {log.name}")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"simulator initialization timed out: run={run}, log={log.name}")
            time.sleep(0.2)
    except BaseException:
        stop_owned_process(process)
        raise


def stop_owned_process(process):
    """Only the newly spawned simulator session belongs to this launcher."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def attach_directory(run):
    directory = (run / "artifacts").resolve()
    manifest = rgb.read_json(directory / "manifest.json")
    args = Path(f"/proc/{manifest['pid']}/cmdline").read_bytes().split(b"\0")
    if str(ROOT / "scripts/codex_symbolic_eval.py").encode() not in args or str(run.resolve()).encode() not in args:
        raise ValueError("original simulator process is gone or PID was reused; never restart this episode")
    if rgb.read_json(directory / "result.json")["lifecycle"] == "closed":
        raise ValueError("cannot resume a closed physical episode")
    return directory


def review_directory(run):
    directory = (run / "artifacts").resolve()
    if rgb.read_json(directory / "result.json")["lifecycle"] != "closed":
        raise ValueError("--resume-review requires a closed episode")
    if not (directory / "agent/conversation.json").exists():
        raise ValueError("review recovery requires the original API conversation")
    return directory


def apply_pending_launch_settings(args):
    """Read operator settings once for a fresh episode; never change attached/live episodes."""
    if args.run_dir is None or args.attach or args.resume_review:
        return
    path = args.run_dir.parent / "pending-launch-settings.json"
    if not path.exists():
        return
    settings = rgb.read_json(path)
    if set(settings) - {"max_steps", "tasks", "reason", "updated_at"}:
        raise ValueError("unknown pending launch setting")
    if args.task not in settings.get("tasks", [args.task]):
        return
    value = settings.get("max_steps")
    if type(value) is not int or value < 1:
        raise ValueError("pending launch max_steps must be a positive integer")
    args.max_steps = value


def main():
    p = parser()
    args = p.parse_args()
    apply_pending_launch_settings(args)
    if args.max_steps is None:
        args.max_steps = 256 if args.smoke else 1300
    if (
        args.max_calls < 4
        or args.max_steps < 1
        or not 1 <= args.prefix <= 256
        or args.max_context_tokens <= args.max_output_tokens
        or args.max_output_tokens < 128
        or min(args.api_timeout, args.tool_timeout, args.startup_timeout) <= 0
    ):
        p.error("invalid budget, prefix or timeout")
    if args.attach and args.resume_review:
        p.error("--attach and --resume-review are mutually exclusive")
    if not args.attach and not args.resume_review and args.policy_backend == "real" and args.policy_port is None:
        p.error("--policy-port is required for real backend; start the pi service separately")
    if args.policy_port is not None and not 1 <= args.policy_port <= 65535:
        p.error("invalid policy port")
    if args.policy_timeout <= 0:
        p.error("policy-timeout must be positive")
    if args.max_delegated_chunks < 1 or (args.supervision_mode == "jev" and args.jev_config is None):
        p.error("invalid delegation limit or missing JEV config")
    if args.max_scheduled_steps < 30 or args.max_scheduled_steps % 10:
        p.error("max-scheduled-steps must be a multiple of ten, at least 30")
    if args.supervision_mode == "step-budget" and args.jev_config:
        p.error("step-budget is independent of JEV; select --supervision-mode jev to use JEV")
    if args.review_every_chunk and args.supervision_mode != "step-budget":
        p.error("--review-every-chunk requires --supervision-mode step-budget")
    if args.frozen_manifest and args.experience_update:
        p.error("frozen experiments require --no-experience-update")
    process = None

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with create_client(args.env_file, timeout=args.api_timeout, trust_env=args.trust_env) as client:
            if args.resume_review:
                directory = review_directory(args.resume_review)
            elif args.attach:
                directory = attach_directory(args.attach)
            else:
                directory, process = launch(args)
            code = Agent(directory, client, args).run()
        if process is not None:
            process.wait(timeout=15)
    finally:
        if process is not None:
            stop_owned_process(process)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
