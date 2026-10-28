"""Persistent, file-based RGB/proprioception bridge for a human or Codex driving RoboMME.

The simulator is trusted to generate task demonstrations and judge success. Only the
allowlisted observations leave that process; neither rewards nor oracle annotations
are recorded. The client requires only Python's standard library.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import UTC
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "runs/evaluation/codex_rgb"
STATE_KEYS = ("joint_state_list", "eef_state_list", "gripper_state_list")
TASKS = (
    "PickXtimes",
    "StopCube",
    "SwingXtimes",
    "BinFill",
    "VideoUnmaskSwap",
    "VideoUnmask",
    "ButtonUnmaskSwap",
    "ButtonUnmask",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "PickHighlight",
    "InsertPeg",
    "MoveCube",
    "PatternLock",
    "RouteStick",
)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def append_json(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def last_scalar(value: object, default: object = None) -> object:
    if value is None:
        return default
    if isinstance(value, list | tuple):
        return last_scalar(value[-1], default) if value else default
    if hasattr(value, "numel") and value.numel() == 0:
        return default
    if hasattr(value, "item"):
        return value.item()
    return value


def safe_feedback(info: dict, *, terminated: object = False, truncated: object = False) -> dict:
    """Do not copy info, reward, error messages, or subgoal labels into any artifact."""
    status = last_scalar(info.get("status"), "ongoing")
    return {
        "success": bool(last_scalar(info.get("success"), default=False)) or status == "success",
        "terminated": bool(last_scalar(terminated, default=False)) or status == "fail",
        "truncated": bool(last_scalar(truncated, default=False)) or status == "timeout",
        "execution_error": status == "error",
    }


def outcome(feedback: dict) -> str:
    if feedback["execution_error"]:
        return "error"
    if feedback["success"]:
        return "success"
    if feedback["truncated"]:
        return "timeout"
    if feedback["terminated"]:
        return "failure"
    return "incomplete"


_IK_ERROR = re.compile(r"^ee step \((?:rpy|quat)\): IK failed \(status=(.*), num_solutions=(\d+)\)$")
_IK_DISTANCE = re.compile(r"^IK Failed! Distance ([0-9.eE+-]+) is greater than threshold ([0-9.eE+-]+)\.$")


def recoverable_control_error(info: dict) -> dict | None:
    """Return a safe, actionable description for errors that do not corrupt the episode."""
    if last_scalar(info.get("status"), "ongoing") != "error":
        return None
    message = last_scalar(info.get("error_message"))
    if not isinstance(message, str):
        return None
    match = _IK_ERROR.fullmatch(message)
    if match is None:
        return None
    status, num_solutions = match.groups()
    result = {"type": "ik_failed", "retryable": True, "num_solutions": int(num_solutions)}
    distance = _IK_DISTANCE.fullmatch(status)
    if distance is not None:
        result.update(distance=float(distance.group(1)), threshold=float(distance.group(2)))
    return result


def validate_action(request: dict) -> tuple[list[float], int, int]:
    target = request.get("target")
    if not isinstance(target, list) or len(target) != 7:
        raise ValueError("target must contain exactly 7 numbers: xyz, rpy, gripper")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in target):
        raise ValueError("target must contain finite JSON numbers, not strings or booleans")
    if not -1 <= target[6] <= 1:
        raise ValueError("gripper must be in [-1, 1]; -1 closes and +1 opens")
    steps = request.get("steps", 1)
    interpolate = request.get("interpolate", 0)
    if type(steps) is not int or not 1 <= steps <= 100:
        raise ValueError("steps must be an integer in [1, 100]")
    if type(interpolate) is not int or not 0 <= interpolate <= steps:
        raise ValueError("interpolate must be an integer in [0, steps]")
    return [float(x) for x in target], steps, interpolate


def action_at(start: list[float], target: list[float], index: int, interpolate: int) -> list[float]:
    if not interpolate:
        return target.copy()
    fraction = min((index + 1) / interpolate, 1.0)
    position = [a + fraction * (b - a) for a, b in zip(start[:3], target[:3], strict=True)]
    angles = [a + fraction * math.remainder(b - a, 2 * math.pi) for a, b in zip(start[3:6], target[3:6], strict=True)]
    return position + angles + [target[6]]


@contextmanager
def quiet_simulator():
    """Suppress Python and native simulator output: it may contain privileged labels."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    with open(os.devnull, "w") as sink:
        try:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd, original in zip((1, 2), saved, strict=True):
                os.dup2(original, fd)
                os.close(original)


class Recorder:
    def __init__(self, directory: Path, fps: float):
        import imageio.v2 as imageio

        self.directory = directory
        self.frame_count = 0
        self.latest = {}
        self.video_context = None
        self.video = imageio.get_writer(
            str(directory / "rollout.mp4"), fps=fps, codec="libx264", quality=8, macro_block_size=1
        )

    def capture(self, obs: dict | None, info: dict, batch: str, feedback: dict) -> None:
        import numpy as np
        from PIL import Image
        from PIL import ImageDraw

        obs = obs or {}
        front_frames = obs.get("front_rgb_list", [])
        count = len(front_frames)
        if any(len(obs.get(key, [])) != count for key in ("wrist_rgb_list", *STATE_KEYS)):
            raise ValueError("unaligned public observation sequences")
        folder = self.directory / "observations" / batch
        folder.mkdir(parents=True, exist_ok=False)
        sheets = []
        sheet = None
        first = self.frame_count
        for i in range(count):
            frame_id = self.frame_count
            paths = {}
            images = []
            for camera in ("front", "wrist"):
                pixels = np.asarray(obs[f"{camera}_rgb_list"][i], dtype=np.uint8)
                img = Image.fromarray(pixels)
                path = folder / f"{frame_id:07d}_{camera}.png"
                img.save(path)
                paths[camera] = str(path.relative_to(self.directory))
                images.append(img)
            panel_height = max(img.height for img in images)
            images = [img.resize((round(img.width * panel_height / img.height), panel_height)) for img in images]
            combined = Image.new("RGB", (sum(img.width for img in images), panel_height + 24))
            offset = 0
            for img in images:
                combined.paste(img, (offset, 24))
                offset += img.width
            ImageDraw.Draw(combined).text((4, 4), f"frame {frame_id} | {batch} | front / wrist", fill="white")
            if self.video_context is not None:
                from scripts.codex_supervisor.video import labeled_frame

                task = str(last_scalar(info.get("task_goal"), self.latest.get("task_goal", "")))
                video_frame = labeled_frame(images[0], images[1], frame_id, self.video_context, task)
            else:
                video_frame = combined
            self.video.append_data(np.asarray(video_frame))
            if i % 12 == 0:
                sheet = Image.new("RGB", (384 * 3, 240 * 4))
            tile = combined.copy()
            tile.thumbnail((384, 216))
            x, y = (i % 3) * 384, ((i % 12) // 3) * 240
            sheet.paste(tile, (x, y + 20))
            ImageDraw.Draw(sheet).text((x + 3, y + 3), f"frame {frame_id}", fill="white")
            if i % 12 == 11 or i == count - 1:
                path = folder / f"sheet_{i // 12:04d}.jpg"
                sheet.save(path, quality=90)
                sheets.append(str(path.relative_to(self.directory)))
            state = {key.removesuffix("_list"): np.asarray(obs[key][i]).reshape(-1).tolist() for key in STATE_KEYS}
            record = {"frame_id": frame_id, "batch": batch, "rgb": paths, "robot": state}
            if self.video_context is not None:
                record["execution"] = {**self.video_context, "task": task}
            append_json(self.directory / "frames.jsonl", record)
            self.latest = record
            self.frame_count += 1
        goal = last_scalar(info.get("task_goal"), self.latest.get("task_goal", ""))
        self.latest = {
            **self.latest,
            "task_goal": str(goal),
            "feedback": feedback,
            "last_batch": f"observations/{batch}/batch.json",
        }
        write_json(
            folder / "batch.json",
            {
                "batch": batch,
                "first_frame": first if count else None,
                "frame_count": count,
                "contact_sheets": sheets,
                "feedback": feedback,
                "note": "reset contains official task-conditioning history followed by the initial observation"
                if batch == "reset"
                else "all frames returned by the simulator are preserved",
            },
        )
        write_json(self.directory / "latest.json", self.latest)

    def close(self) -> None:
        self.video.close()


def build_environment(args: argparse.Namespace) -> tuple[object, dict]:
    # Use the simulator revision in this repository, even with a shared Python env.
    sys.path.insert(0, str(ROOT / "third_party/robomme_benchmark/src"))
    import gymnasium as gym
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    from robomme.env_record_wrapper import DemonstrationWrapper
    from robomme.env_record_wrapper import EndeffectorDemonstrationWrapper
    from robomme.env_record_wrapper import FailAwareWrapper
    import robomme.robomme_env  # Registers benchmark tasks.

    builder = BenchmarkEnvBuilder(args.task, dataset=args.dataset, action_space="ee_pose")
    if (args.task, args.episode) not in builder.metadata_index:
        raise ValueError("episode is absent from the selected official split")
    metadata_seed, difficulty = builder.resolve_episode(args.episode)
    seed = args.seed if args.seed is not None else metadata_seed
    kwargs = {
        "obs_mode": "rgb+depth+segmentation",
        "control_mode": "pd_joint_pos",
        "render_mode": "rgb_array",
        "reward_mode": "dense",
        "seed": seed,
        "sim_backend": args.sim_backend,
    }
    if difficulty:
        kwargs["difficulty"] = difficulty
    if args.render_backend:
        kwargs["render_backend"] = args.render_backend
    # Matches the official builder. Internal depth/segmentation remain inside the
    # trusted benchmark (task rendering/judging); all optional outputs stay disabled.
    env = gym.make(args.task, **kwargs)
    env = DemonstrationWrapper(env, max_steps_without_demonstration=args.max_steps + 2, gui_render=False)
    env = FailAwareWrapper(EndeffectorDemonstrationWrapper(env, action_repr="rpy"))
    import robomme

    return env, {
        "seed": seed,
        "difficulty": difficulty,
        "robomme_module": str(Path(robomme.__file__).resolve()),
        "control_frequency_hz": float(env.unwrapped.control_freq),
    }


def serve(args: argparse.Namespace) -> None:
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(OUTPUT_ROOT.resolve()) or directory == OUTPUT_ROOT.resolve():
        raise ValueError(f"run-dir must be a new directory below {OUTPUT_ROOT}")
    directory.mkdir(parents=True, exist_ok=False)
    for name in ("requests", "responses", "observations"):
        (directory / name).mkdir()
    (directory / "memory.md").write_text("# Episode memory\n\nRecord observed evidence with frame IDs here.\n")
    manifest = {
        "schema_version": 1,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task": args.task,
        "dataset": args.dataset,
        "episode": args.episode,
        "seed_override": args.seed,
        "max_steps": args.max_steps,
        "fps": args.fps,
        "sim_backend": args.sim_backend,
        "render_backend": args.render_backend,
        "action_space": "ee_pose",
        "python": sys.executable,
        "pid": os.getpid(),
        "started_at": datetime.now(UTC).isoformat(),
        "experience_sha256": None,
    }
    if args.experience:
        content = args.experience.read_bytes()
        (directory / "experience-input.md").write_bytes(content)
        manifest["experience_sha256"] = hashlib.sha256(content).hexdigest()
    write_json(directory / "manifest.json", manifest)
    feedback = safe_feedback({})
    state = {
        "lifecycle": "initializing",
        "outcome": "incomplete",
        "feedback": feedback,
        "action_requests": 0,
        "control_steps": 0,
        "recorded_frames": 0,
        "video_finalized": False,
    }
    write_json(directory / "result.json", state)
    env = recorder = None
    pending = None
    stop = False

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        with quiet_simulator():
            env, resolved = build_environment(args)
            write_json(directory / "manifest.json", {**manifest, **resolved})
            recorder = Recorder(directory, args.fps)
            obs, info = env.reset()
            feedback = safe_feedback(info)
            recorder.capture(obs, info, "reset", feedback)
        state.update(lifecycle="ready", feedback=feedback, recorded_frames=recorder.frame_count)
        write_json(directory / "result.json", state)
        print(json.dumps({"ready": str(directory), "latest": str(directory / "latest.json")}), flush=True)
        while not stop and outcome(feedback) == "incomplete":
            requests = sorted((directory / "requests").glob("*.json"))
            for path in requests:
                response_path = directory / "responses" / path.name
                if response_path.exists():
                    continue
                pending = response_path
                request = read_json(path)
                operation = request.get("operation")
                if operation == "finish":
                    stop = True
                    break
                if operation != "act":
                    write_json(response_path, {"ok": False, "error": "unknown operation"})
                    pending = None
                    continue
                if request.get("expected_frame") != recorder.latest.get("frame_id"):
                    write_json(response_path, {"ok": False, "error": "stale observation; inspect latest.json"})
                    pending = None
                    continue
                try:
                    target, steps, interpolate = validate_action(request)
                except ValueError as exc:
                    write_json(response_path, {"ok": False, "error": str(exc)})
                    pending = None
                    continue
                state["action_requests"] += 1
                append_json(
                    directory / "actions.jsonl",
                    {
                        "request_id": path.stem,
                        "based_on_frame": recorder.latest.get("frame_id"),
                        "target": target,
                        "steps": steps,
                        "interpolate": interpolate,
                    },
                )
                start = recorder.latest["robot"]["eef_state"]
                state["lifecycle"] = "executing"
                state.pop("last_control_error", None)
                write_json(directory / "result.json", state)
                executed = 0
                control_error = None
                for i in range(steps):
                    if stop:
                        break
                    action = action_at(start, target, i, interpolate)
                    append_json(
                        directory / "controls.jsonl",
                        {
                            "request_id": path.stem,
                            "control_step": state["control_steps"],
                            "action": action,
                        },
                    )
                    step_index = state["control_steps"]
                    state["control_steps"] += 1
                    with quiet_simulator():
                        obs, _reward, terminated, truncated, info = env.step(action)
                        control_error = recoverable_control_error(info)
                        if control_error is None:
                            feedback = safe_feedback(info, terminated=terminated, truncated=truncated)
                        else:
                            # An IK miss happens before the inner environment step. Stop this
                            # request, preserve the episode and let the controller choose a
                            # smaller or different target from the last valid observation.
                            feedback = {**safe_feedback({}), "control_error": control_error}
                        recorder.capture(obs, info, f"step_{step_index:07d}", feedback)
                    executed += 1
                    if control_error is not None:
                        state["last_control_error"] = control_error
                        break
                    if outcome(feedback) != "incomplete":
                        break
                    if state["control_steps"] >= args.max_steps:
                        state["stop_reason"] = "control_budget"
                        stop = True
                        break
                state.update(
                    lifecycle="ready",
                    feedback=feedback,
                    outcome=outcome(feedback),
                    recorded_frames=recorder.frame_count,
                )
                write_json(directory / "result.json", state)
                response = {"ok": True, "executed_steps": executed, "result": state, "latest": recorder.latest}
                if control_error is not None:
                    response["control_error"] = control_error
                write_json(response_path, response)
                pending = None
                if stop or outcome(feedback) != "incomplete":
                    break
            if not stop and outcome(feedback) == "incomplete":
                time.sleep(0.05)
    except Exception as exc:
        # Exception text / locals / traceback may embed oracle state. Publish only its type.
        feedback = {**feedback, "execution_error": True}
        state["error_type"] = type(exc).__name__
        state["error_location"] = [
            {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
            for frame in traceback.extract_tb(exc.__traceback__)
        ]
    finally:
        if recorder is not None:
            try:
                with quiet_simulator():
                    recorder.close()
                state["video_finalized"] = recorder.frame_count > 0
            except Exception as exc:
                feedback = {**feedback, "execution_error": True}
                state["video_error_type"] = type(exc).__name__
            state["recorded_frames"] = recorder.frame_count
        if env is not None:
            try:
                with quiet_simulator():
                    env.close()
            except Exception as exc:
                state["close_error_type"] = type(exc).__name__
        state.update(
            lifecycle="closed", outcome=outcome(feedback), feedback=feedback, ended_at=datetime.now(UTC).isoformat()
        )
        write_json(directory / "result.json", state)
        if pending is not None:
            write_json(pending, {"ok": not feedback["execution_error"], "result": state})
    print(json.dumps(state), flush=True)
    if feedback["execution_error"]:
        raise SystemExit(1)


def submit(args: argparse.Namespace) -> None:
    directory = args.run_dir.resolve()
    state = read_json(directory / "result.json")
    if state["lifecycle"] == "closed":
        print(json.dumps({"ok": args.command == "finish", "result": state}, indent=2))
        return
    if state["lifecycle"] != "ready":
        raise ValueError("server is not ready; inspect result.json before submitting")
    request = {"operation": args.command}
    if args.command == "act":
        request.update(target=args.target, steps=args.steps, interpolate=args.interpolate)
        validate_action(request)
        request["expected_frame"] = read_json(directory / "latest.json").get("frame_id")
    request_id = f"{time.time_ns()}_{uuid.uuid4().hex}"
    write_json(directory / "requests" / f"{request_id}.json", request)
    response = directory / "responses" / f"{request_id}.json"
    deadline = time.monotonic() + args.timeout
    while not response.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"request {request_id} is still pending; inspect {response}; DO NOT resend the action")
        time.sleep(0.1)
    print(response.read_text())


def verify(args: argparse.Namespace) -> None:
    """Decode the finished video and validate frame archives without importing RoboMME."""
    import imageio.v2 as imageio
    from PIL import Image

    directory = args.run_dir.resolve()
    state = read_json(directory / "result.json")
    if state["lifecycle"] != "closed" or not state["video_finalized"]:
        raise ValueError("run must be closed with a finalized video before verification")
    frame_count = 0
    with (directory / "frames.jsonl").open() as handle:
        for line in handle:
            frame = json.loads(line)
            if frame["frame_id"] != frame_count:
                raise ValueError("frame IDs are not contiguous")
            for camera in ("front", "wrist"):
                path = (directory / frame["rgb"][camera]).resolve()
                if not path.is_relative_to(directory):
                    raise ValueError("image path escapes run directory")
                with Image.open(path) as img:
                    img.verify()
            frame_count += 1
    with imageio.get_reader(directory / "rollout.mp4") as reader:
        video_frames = sum(1 for _ in reader)
    if not frame_count == video_frames == state["recorded_frames"]:
        raise ValueError("recorded frames, archived frames and decoded video frame count differ")
    print(
        json.dumps(
            {
                "verified": True,
                "archived_frames": frame_count,
                "decoded_video_frames": video_frames,
                "outcome": state["outcome"],
                "video": str(directory / "rollout.mp4"),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="keep one episode alive until finish or environment termination")
    server.add_argument("--run-dir", type=Path, required=True)
    server.add_argument("--task", choices=TASKS, default="PickXtimes")
    server.add_argument("--dataset", choices=("train", "test", "val"), default="test")
    server.add_argument("--episode", type=int, default=0)
    server.add_argument("--seed", type=int, help="override official episode seed; recorded as a custom configuration")
    server.add_argument("--max-steps", type=int, default=1300)
    server.add_argument("--fps", type=float, default=20, help="playback FPS; wall-clock reasoning pauses are excluded")
    server.add_argument("--sim-backend", default="cpu")
    server.add_argument("--render-backend", default=os.environ.get("ROBOMME_RENDER_BACKEND"))
    server.add_argument("--experience", type=Path, help="snapshot prior evidence-based experience into this run")
    server.set_defaults(func=serve)
    for name in ("act", "finish", "status", "verify"):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "status":
            command.set_defaults(func=lambda args: print((args.run_dir / "result.json").read_text()))
        elif name == "verify":
            command.set_defaults(func=verify)
        else:
            command.add_argument("--timeout", type=float, default=120)
            command.set_defaults(func=submit)
        if name == "act":
            command.add_argument("--target", type=float, nargs=7, required=True, metavar="N")
            command.add_argument("--steps", type=int, default=1)
            command.add_argument("--interpolate", type=int, default=0)
    args = parser.parse_args()
    if args.command == "serve" and (args.max_steps < 1 or not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("max-steps and fps must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
