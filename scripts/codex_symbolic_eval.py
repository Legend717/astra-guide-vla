"""Current-session GPT supervising a paused symbolic VLA episode via auditable files."""

from __future__ import annotations

import argparse
from datetime import UTC
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import codex_rgb_eval as rgb  # noqa: E402
from scripts.codex_supervisor import experience  # noqa: E402
from scripts.codex_supervisor.instructions import annotate  # noqa: E402
from scripts.codex_supervisor.instructions import convert  # noqa: E402
from scripts.codex_supervisor.layout import artifacts  # noqa: E402
from scripts.codex_supervisor.layout import create_run  # noqa: E402

OUTPUT_ROOT = ROOT / "runs/evaluation/codex_symbolic"
DEFAULT_CHECKPOINT = ROOT / "runs/ckpts/mme_vla_suite/symbolic-grounded-subgoal/79999"


def preflight(args):
    """Check both real interpreters and GPU without loading weights or resetting a task."""
    checkpoint = args.checkpoint.resolve()
    for path in (
        checkpoint / "params",
        checkpoint / "assets/robomme/norm_stats.json",
        checkpoint.parent / "history_config.txt",
    ):
        if not path.exists():
            raise ValueError(f"missing checkpoint component: {path}")
    expected = "symbolic-simple-subgoal.yaml" if args.mode == "simple-smoke" else "symbolic-grounded-subgoal.yaml"
    if (checkpoint.parent / "history_config.txt").read_text().strip() != expected:
        raise ValueError("checkpoint history_config.txt does not match mode")
    probe = """
import importlib, json, sys
for name in sys.argv[1:]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        print(json.dumps({'ok': False, 'module': name, 'error_type': type(exc).__name__}))
        sys.exit(1)
if 'jax' in sys.argv[1:]:
    devices = [str(device) for device in module.devices() if device.platform == 'gpu']
    print(json.dumps({'ok': bool(devices), 'python': sys.executable, 'gpu_devices': devices}))
    sys.exit(0 if devices else 1)
print(json.dumps({'ok': True, 'python': sys.executable}))
"""
    reports = {}
    for name, executable, modules in (
        ("simulator", sys.executable, ["mani_skill", "sapien", "robomme.env_record_wrapper"]),
        # Resolving the venv python symlink would bypass pyvenv.cfg and its installed packages.
        ("policy", str(args.policy_python.absolute()), ["mme_vla_suite", "jax"]),
    ):
        result = subprocess.run([executable, "-c", probe, *modules], capture_output=True, text=True, check=False)
        try:
            report = json.loads(result.stdout.splitlines()[-1])
        except (ValueError, IndexError):
            report = {"ok": False, "error": "dependency probe did not return JSON", "returncode": result.returncode}
        reports[name] = report
    print(json.dumps({"checkpoint": str(checkpoint), **reports}, indent=2))
    if not all(report.get("ok") for report in reports.values()):
        raise SystemExit(
            "Preflight failed. Check interpreter/dependencies; GPU visibility may require sandbox escalation."
        )


def history(args):
    from scripts.codex_supervisor.history import render_history

    directory = artifacts(args.run_dir)
    last = args.to_frame
    if last is None:
        last = rgb.read_json(directory / "latest.json")["frame_id"]
    first = args.from_frame
    if args.request_id:
        if first is not None or args.to_frame is not None:
            raise ValueError("choose request-id or an explicit frame range")
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        relevant = [event for event in events if event.get("request_id") == args.request_id]
        starts = [event for event in relevant if event["event"] in ("release", "intervene")]
        ends = [event for event in relevant if event["event"] == "execution_end"]
        if len(starts) != 1 or len(ends) != 1:
            raise ValueError("request must identify one completed release/intervene")
        first, last = starts[0]["frame"] + 1, ends[0]["after_frame"]
        if first > last:
            print(json.dumps({"frame_count": 0, "contact_sheets": [], "note": "No new returned frames."}))
            return
    print(json.dumps(render_history(directory, last if first is None else first, last), indent=2))


def immutable_request(directory, identifier, request):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", identifier):
        raise ValueError("request ID must use 1-100 letters, digits, underscores or hyphens")
    path = directory / "requests" / f"{identifier}.json"
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    rgb.write_json(temporary, request)
    try:
        try:
            os.link(temporary, path)
            rgb.append_json(directory / "inputs.jsonl", {"request_id": identifier, "request": request})
        except FileExistsError:
            if rgb.read_json(path) != request:
                raise ValueError("request ID already exists with a different body") from None
    finally:
        temporary.unlink()
    return directory / "responses" / path.name


def serve(args):
    from scripts.codex_supervisor.core import Supervisor
    from scripts.codex_supervisor.kinematics import RobotFK
    from scripts.codex_supervisor.runtime import Runtime
    from scripts.codex_supervisor.runtime import WorkerPolicy

    directory = args.run_dir.resolve()
    output_root = getattr(args, "output_root", OUTPUT_ROOT).resolve()
    if not directory.is_relative_to(output_root) or directory == output_root:
        raise ValueError(f"run-dir must be new and below {output_root}")
    directory = create_run(directory)
    for name in ("requests", "responses", "observations", "internal"):
        (directory / name).mkdir()
    (directory / "memory.md").write_text(
        "# Episode memory\n\nRecord legal observations and frame IDs; distinguish hypotheses.\n"
    )
    (directory / "review.md").write_text(
        "# Review\n\nPending. Record official outcome separately from harness verification.\n"
    )
    manifest = {
        "schema_version": 2,
        "observation_boundary": "executed RGB history and robot state; pending FK numbers allowed, no future images or trial execution",
        "pid": os.getpid(),
        "task": args.task,
        "dataset": args.dataset,
        "episode": args.episode,
        "seed_override": args.seed,
        "max_steps": args.max_steps,
        "prefix": args.prefix,
        "mode": args.mode,
        "client_id": args.client_id,
        "fps": args.fps,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "checkpoint": (
            str(args.checkpoint.resolve())
            if getattr(args, "policy_backend", "real") == "real" and getattr(args, "policy_port", None) is None
            else None
        ),
        "policy_endpoint": (
            f"ws://{args.policy_host}:{args.policy_port}" if getattr(args, "policy_port", None) is not None else None
        ),
        "policy_backend": getattr(args, "policy_backend", "real"),
        "config": args.config,
        "policy_python": str(args.policy_python),
        "python": sys.executable,
        "started_at": datetime.now(UTC).isoformat(),
        "experience_input": experience.snapshot(directory, args.experience, enabled=args.experience_input),
        "experience_update": args.experience_update,
        "experience_update_policy": getattr(args, "experience_update_policy", "always"),
    }
    if getattr(args, "task_experience", None) is not None:
        manifest["task_experience_input"] = experience.snapshot(
            directory, args.task_experience, enabled=args.experience_input, filename="experience-task-input.json"
        )
    rgb.write_json(directory / "manifest.json", manifest)
    rgb.write_json(directory / "result.json", {"lifecycle": "initializing", "pid": os.getpid()})
    env = recorder = policy = supervisor = None
    pending = None
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
        if supervisor is not None:
            supervisor.closed = True
            supervisor.state["stop_reason"] = "signal"

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    error = None
    try:
        with rgb.quiet_simulator():
            env, resolved = rgb.build_environment(args)
            runtime = Runtime(env)
            obs, info = runtime.reset()
            recorder = rgb.Recorder(directory, args.fps)
            recorder.video_context = {"source": "RESET", "request_id": "reset", "subgoal": "(not set)"}
            feedback = rgb.safe_feedback(info)
            recorder.capture(obs, info, "reset", feedback)
            fk = RobotFK(env)
        manifest.update(resolved, fk=fk.metadata)
        rgb.write_json(directory / "manifest.json", manifest)
        if getattr(args, "policy_backend", "real") == "mock-stall":
            from scripts.codex_supervisor.mock_policy import StallPolicy

            policy = StallPolicy()
        elif getattr(args, "policy_port", None) is not None:
            from scripts.codex_supervisor.remote_policy import RemotePolicy

            policy = RemotePolicy(
                args.policy_host, args.policy_port, timeout=args.policy_timeout, mode=args.mode, seed=args.model_seed
            )
        else:
            policy = WorkerPolicy(directory, args)
        manifest["policy"] = policy.metadata
        rgb.write_json(directory / "manifest.json", manifest)
        supervisor = Supervisor(
            directory,
            recorder,
            runtime,
            policy,
            fk,
            task=args.task,
            mode=args.mode,
            max_steps=args.max_steps,
            prefix=args.prefix,
        )
        supervisor.state.update(feedback=feedback, outcome=rgb.outcome(feedback))
        supervisor.closed = rgb.outcome(feedback) != "incomplete"
        supervisor.publish()
        print(json.dumps({"ready": str(directory), "pid": os.getpid()}), flush=True)
        while not stopped and not supervisor.closed:
            for path in sorted((directory / "requests").glob("*.json")):
                response = directory / "responses" / path.name
                if response.exists():
                    continue
                pending = response
                request = rgb.read_json(path)
                if request.get("client_id") != args.client_id:
                    rgb.write_json(response, {"ok": False, "error": "wrong controlling client"})
                    pending = None
                    continue
                rgb.append_json(
                    directory / "events.jsonl",
                    {
                        "event": "request_started",
                        "request_id": path.stem,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    },
                )
                try:
                    with rgb.quiet_simulator():
                        result = supervisor.handle(request, path.stem)
                except ValueError as exc:
                    if supervisor.state["lifecycle"] != "ready":
                        raise
                    result = {"ok": False, "error": str(exc)}  # Only controlled protocol validation errors.
                supervisor.publish()
                rgb.write_json(response, {**result, "request_id": path.stem, "result": supervisor.state})
                pending = None
                if supervisor.closed:
                    break
            time.sleep(0.05)
    except Exception as exc:
        error = type(exc).__name__
        rgb.write_json(
            directory / "error-location.json",
            [
                {"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                for f in traceback.extract_tb(exc.__traceback__)
            ],
        )
    finally:
        state = (
            supervisor.state
            if supervisor
            else {
                "control_steps": 0,
                "inference_count": 0,
                "recorded_frames": 0,
                "feedback": rgb.safe_feedback({}),
                "outcome": "incomplete",
                "video_finalized": False,
            }
        )
        for resource in (recorder, env, policy):
            state.setdefault("physical_ended_epoch", time.time())
            if resource is not None:
                try:
                    with rgb.quiet_simulator():
                        resource.close()
                    if resource is recorder:
                        state["video_finalized"] = recorder.frame_count > 0
                except Exception as exc:
                    error = type(exc).__name__
        if recorder:
            state["recorded_frames"] = recorder.frame_count
        if error:
            state.update(outcome="error", error_type=error)
            state["feedback"]["execution_error"] = True
        state.update(lifecycle="closed", candidate_id=None, ended_at=datetime.now(UTC).isoformat())
        if supervisor:
            supervisor.candidate = supervisor.actions = None
            supervisor.publish()
        rgb.write_json(directory / "result.json", state)
        if pending:
            rgb.write_json(pending, {"ok": False, "result": state})
    print(json.dumps(state), flush=True)
    if error:
        raise SystemExit(1)


def submit(args):
    directory = artifacts(args.run_dir)
    if args.request_json is not None:
        request = json.loads(args.request_json)
    elif str(args.request) == "-":
        request = json.load(sys.stdin)
    else:
        request = rgb.read_json(args.request)
    request["client_id"] = args.client_id
    # A process lock serializes submissions; one stable logical owner is pinned by serve.
    with (directory / "client.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        response = immutable_request(directory, args.request_id, request)
        print(json.dumps({"request_id": args.request_id, "response": str(response)}), flush=True)
        deadline = time.monotonic() + args.timeout
        while not response.exists():
            if time.monotonic() >= deadline:
                print(json.dumps({"pending": True, "note": "Timeout is not cancellation. Query this request ID."}))
                return
            if rgb.read_json(directory / "result.json")["lifecycle"] == "closed":
                print(
                    json.dumps({"pending": True, "note": "Server closed; inspect result. Do not restart this episode."})
                )
                return
            time.sleep(0.1)
        print(response.read_text())


def inspect(args):
    directory = artifacts(args.run_dir)
    latest = rgb.read_json(directory / "latest.json")
    from PIL import Image

    image_path = directory / latest["rgb"]["front"]
    with Image.open(image_path) as image:
        size = (image.height, image.width)
    manifest = rgb.read_json(directory / "manifest.json")
    converted = convert(manifest["task"], rgb.read_json(args.subgoal), size)
    print(json.dumps({"expected_frame": latest["frame_id"], **converted}, indent=2))
    if args.mark:
        (directory / "inspection").mkdir(exist_ok=True)
        path = directory / "inspection" / f"points-frame-{latest['frame_id']}-{uuid.uuid4().hex}.png"
        annotate(image_path, path, converted["pixel_yx"])
        print(path)


def verify(args):
    args = argparse.Namespace(**{**vars(args), "run_dir": artifacts(args.run_dir)})
    rgb.verify(args)
    directory = args.run_dir.resolve()
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    result = rgb.read_json(directory / "result.json")
    controls = [e for e in events if e["event"] == "control_return"]
    if len(controls) != result["control_steps"]:
        raise ValueError("some submitted controls have no recorded return; inspect interrupted run")
    candidates = [e for e in events if e["event"] == "candidate"]
    if len(candidates) != result["inference_count"]:
        raise ValueError("inference/candidate count mismatch")
    from scripts.codex_supervisor.audit import verify_events

    verify_events(directory, events, result)
    manifest = rgb.read_json(directory / "manifest.json")
    snapshot = manifest["experience_input"]
    if snapshot["enabled"]:
        digest = hashlib.sha256((directory / "experience-input.json").read_bytes()).hexdigest()
        if digest != snapshot["sha256"]:
            raise ValueError("experience input snapshot changed")
    print(json.dumps({"supervisor_verified": True, "candidates": len(candidates), "controls": len(controls)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve")
    server.add_argument("--run-dir", type=Path, required=True)
    server.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    server.add_argument("--client-id", required=True)
    server.add_argument("--task", choices=rgb.TASKS, required=True)
    server.add_argument("--dataset", choices=("train", "test", "val"), default="test")
    server.add_argument("--episode", type=int, default=0)
    server.add_argument("--seed", type=int)
    server.add_argument("--max-steps", type=int, default=1300)
    server.add_argument("--prefix", type=int, default=16)
    server.add_argument("--fps", type=float, default=20)
    server.add_argument("--sim-backend", default="cpu")
    server.add_argument("--render-backend", default=os.environ.get("ROBOMME_RENDER_BACKEND"))
    server.add_argument("--mode", choices=("grounded", "simple-smoke"), default="grounded")
    server.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    server.add_argument("--config", default="mme_vla_suite")
    server.add_argument("--policy-python", type=Path, default=ROOT / ".venv/bin/python")
    server.add_argument("--policy-host", default="127.0.0.1")
    server.add_argument("--policy-port", type=int)
    server.add_argument("--policy-timeout", type=float, default=180)
    server.add_argument("--model-seed", type=int, default=42)
    server.add_argument("--task-experience", type=Path)
    server.add_argument("--policy-backend", choices=("real", "mock-stall"), default="real")
    server.add_argument("--experience", type=Path, default=OUTPUT_ROOT / "experience/general.json")
    server.add_argument("--experience-input", action=argparse.BooleanOptionalAction, default=True)
    server.add_argument("--experience-update", action=argparse.BooleanOptionalAction, default=True)
    server.add_argument("--experience-update-policy", choices=("always", "failure-only"), default="always")
    server.set_defaults(func=serve)
    check = commands.add_parser("preflight")
    check.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    check.add_argument("--policy-python", type=Path, default=ROOT / ".venv/bin/python")
    check.add_argument("--mode", choices=("grounded", "simple-smoke"), default="grounded")
    check.set_defaults(func=preflight)
    for name in ("submit", "status", "query", "inspect", "verify", "experience-commit", "history", "render-video"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "render-video":
            from scripts.codex_supervisor.video import render_archive

            command.set_defaults(func=lambda a: print(json.dumps(render_archive(artifacts(a.run_dir)))))
        elif name == "history":
            command.add_argument("--from-frame", type=int)
            command.add_argument("--to-frame", type=int)
            command.add_argument("--request-id")
            command.set_defaults(func=history)
        elif name == "submit":
            command.add_argument("--client-id", required=True)
            command.add_argument("--request-id", required=True)
            source = command.add_mutually_exclusive_group(required=True)
            source.add_argument("--request", type=Path, help="JSON path, or - to read stdin")
            source.add_argument("--request-json", help="inline JSON; archived automatically in inputs.jsonl")
            command.add_argument("--timeout", type=float, default=120)
            command.set_defaults(func=submit)
        elif name == "query":
            command.add_argument("--request-id", required=True)
            command.set_defaults(func=query)
        elif name == "inspect":
            command.add_argument("--subgoal", type=Path, required=True)
            command.add_argument("--mark", action="store_true")
            command.set_defaults(func=inspect)
        elif name == "experience-commit":
            command.add_argument("--submission", type=Path, required=True)
            command.set_defaults(
                func=lambda a: print(json.dumps(experience.commit(artifacts(a.run_dir), rgb.read_json(a.submission))))
            )
        elif name == "verify":
            command.set_defaults(func=verify)
        else:
            command.set_defaults(func=lambda a: print((artifacts(a.run_dir) / "result.json").read_text()))
    args = parser.parse_args()
    if args.command == "serve" and (args.max_steps < 1 or not 1 <= args.prefix <= 256 or not 0 < args.fps < 1000):
        parser.error("invalid max-steps, prefix or fps")
    args.func(args)


def query(args):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args.request_id):
        raise ValueError("invalid request ID")
    path = artifacts(args.run_dir) / "responses" / f"{args.request_id}.json"
    print(path.read_text() if path.exists() else json.dumps({"pending": True, "request_id": args.request_id}))


if __name__ == "__main__":
    main()
