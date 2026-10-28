"""Plan the paper's 16-task evaluation; live execution requires --execute."""

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    "BinFill",
    "PickXtimes",
    "SwingXtimes",
    "StopCube",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmask",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)


def validate_experience(directory, tasks, *, initial_preset=False):
    """Reject incomplete or mixed-mode libraries before planning or creating output."""
    counts = {}
    baseline = (
        json.loads((ROOT / "assets/initial-experience/general.json").read_text())["entries"] if initial_preset else None
    )
    for scope in ["general", *tasks]:
        path = directory / ("general.json" if scope == "general" else f"tasks/{scope}.json")
        data = json.loads(path.read_text())
        entries = data["entries"]
        if not isinstance(entries, list) or not isinstance(data.get("version"), int):
            raise ValueError(f"Invalid experience library: {path}")
        ids = [entry.get("id") for entry in entries]
        if any(not isinstance(key, str) or not key for key in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Missing or duplicate experience IDs: {path}")
        for entry in entries:
            if entry.get("applicability") not in {"common", "step-budget"} or "supervision" in entry:
                raise ValueError(f"Non-step-budget experience: {path}")
            if not entry.get("claim") or not entry.get("conditions"):
                raise ValueError(f"Incomplete experience entry: {path}")
            if (
                initial_preset
                and scope != "general"
                and entry.get("provenance", {}).get("kind") != "demonstration_prior"
            ):
                raise ValueError(f"Initial experience must contain training priors only: {path}")
        if initial_preset and scope == "general" and entries != baseline:
            raise ValueError(f"Initial global guidance must match the bundled baseline: {path}")
        if initial_preset and data.get("commits"):
            raise ValueError(f"Initial experience contains learned history: {path}")
        counts[scope] = len(entries)
    return counts


def local_pythonpath(root):
    paths = [root / "src", root / "packages/openpi-client/src", root / "third_party/robomme_benchmark/src"]
    return os.pathsep.join([str(path) for path in paths])


def build_plan(
    root,
    output,
    sim_python,
    host,
    port,
    tasks=None,
    max_steps=1300,
    *,
    experience_updates="protocol",
    episodes=None,
    env_file=None,
    cpu_render_launcher=None,
):
    if experience_updates not in {"protocol", "frozen"}:
        raise ValueError("Unknown experience update mode")
    order = list(TASKS) if tasks is None else list(tasks)
    if len(set(order)) != len(order) or set(order) - set(TASKS):
        raise ValueError("Tasks must be unique RoboMME task names")
    episode_order = list(range(50)) if episodes is None else list(episodes)
    if (
        not episode_order
        or len(set(episode_order)) != len(episode_order)
        or any(not isinstance(episode, int) or not 0 <= episode < 50 for episode in episode_order)
    ):
        raise ValueError("Episodes must be unique integers from 0 to 49")
    plan = []
    for task in order:
        for episode in episode_order:
            command = [
                sys.executable,
                str(root / "scripts/run_robomme_agent.py"),
                "--task",
                task,
                "--episode",
                str(episode),
                "--dataset",
                "test",
                "--run-dir",
                str(output / f"{task.lower()}-ep{episode}"),
                "--sim-python",
                str(sim_python),
                "--policy-backend",
                "real",
                "--policy-host",
                host,
                "--policy-port",
                str(port),
                "--experience",
                str(output / "experience/general.json"),
                "--task-experience",
                str(output / "experience/tasks" / f"{task}.json"),
                "--experience-update" if experience_updates == "protocol" else "--no-experience-update",
                "--experience-update-policy",
                "always" if episode < 5 else "failure-only",
                "--max-steps",
                str(max_steps),
                "--prefix",
                "16",
                "--model",
                "gpt-6-astra",
                "--reasoning",
                "high",
                "--supervision-mode",
                "step-budget",
                "--max-scheduled-steps",
                "200",
                "--max-calls",
                "80",
                "--max-output-tokens",
                "8192",
                "--max-context-tokens",
                "300000",
            ]
            if env_file is not None:
                command.extend(["--env-file", str(env_file)])
            if cpu_render_launcher is not None:
                command.extend(["--cpu-render-launcher", str(cpu_render_launcher)])
            plan.append({"task": task, "episode": episode, "command": command})
    return order, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experience-preset", choices=("initial", "final"), default="initial")
    parser.add_argument("--initial-experience", type=Path, help="Override the preset's experience directory")
    parser.add_argument(
        "--experience-updates",
        choices=("protocol", "frozen"),
        help="Default: protocol for initial priors, frozen for final experience",
    )
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--cpu-render-launcher", type=Path, default=ROOT / "scripts/run_simulator.sh")
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument(
        "--max-steps", type=int, default=1300, help="Execution setting; defaults to the existing agent's 1300 controls"
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--episodes", nargs="+", type=int, help="Episode IDs (0..49); defaults to all 50")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    initial = (args.initial_experience or ROOT / "assets" / f"{args.experience_preset}-experience").resolve()
    updates = args.experience_updates or ("protocol" if args.experience_preset == "initial" else "frozen")
    sim_python = args.sim_python.absolute()
    env_file = args.env_file.absolute()
    launcher = args.cpu_render_launcher.absolute()
    if output.exists():
        parser.error("output must be new; existing experiments are never overwritten")
    if not 1 <= args.policy_port <= 65535 or args.max_steps < 1:
        parser.error("invalid port or control limit")
    order, plan = build_plan(
        ROOT,
        output,
        sim_python,
        args.policy_host,
        args.policy_port,
        args.tasks,
        args.max_steps,
        experience_updates=updates,
        episodes=args.episodes,
        env_file=env_file,
        cpu_render_launcher=launcher,
    )
    try:
        validate_experience(initial, order, initial_preset=args.experience_preset == "initial")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f"Invalid experience: {exc}")
    if not args.execute:
        for item in plan:
            print(shlex.join(item["command"]))
        print(f"{len(plan)} episodes planned; no files written and no services contacted.")
        return
    if not sim_python.is_file():
        parser.error("simulator interpreter does not exist")
    required = [initial / "general.json"]
    required.extend(initial / "tasks" / f"{task}.json" for task in order)
    for path in required:
        if not path.is_file():
            parser.error(f"missing initial experience: {path}")
    output.mkdir(parents=True)
    (output / "experience/tasks").mkdir(parents=True)
    shutil.copyfile(required[0], output / "experience/general.json")
    for task in order:
        shutil.copyfile(initial / "tasks" / f"{task}.json", output / "experience/tasks" / f"{task}.json")
    (output / "experiment.json").write_text(
        json.dumps(
            {
                "experience_preset": args.experience_preset,
                "experience_updates": updates,
                "experience_source": str(initial),
                "tasks": order,
                "episodes": list(dict.fromkeys(item["episode"] for item in plan)),
                "plan": plan,
                "protocol": "serial tasks; shared global experience; separate task experience",
            },
            indent=2,
        )
    )
    env = {**os.environ, "PYTHONPATH": local_pythonpath(ROOT)}
    for item in plan:
        completed = subprocess.run(item["command"], cwd=ROOT, env=env, check=False)
        if completed.returncode:
            raise SystemExit(f"Stopped at {item['task']} episode {item['episode']}; outputs retained.")


if __name__ == "__main__":
    main()
