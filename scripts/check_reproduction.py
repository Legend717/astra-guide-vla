"""Check local reproduction prerequisites without loading weights or running an episode."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_release import TASKS  # noqa: E402
from scripts.evaluate_release import local_pythonpath  # noqa: E402
from scripts.evaluate_release import validate_experience  # noqa: E402


def probe_python(python, modules):
    code = "import importlib, json; modules = " + repr(modules) + "; result = {}\n"
    code += "for name in modules:\n"
    code += " try:\n  m = importlib.import_module(name); result[name] = getattr(m, '__file__', 'built-in')\n"
    code += " except Exception as e:\n  result[name] = {'error': type(e).__name__}\n"
    code += "print(json.dumps(result))\n"
    result = subprocess.run(
        [str(python), "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONPATH": local_pythonpath(ROOT)},
        check=False,
    )
    if result.returncode:
        raise ValueError(f"Interpreter probe exited with code {result.returncode}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--experience-preset", choices=("initial", "final"), default="initial")
    parser.add_argument("--initial-experience", type=Path)
    parser.add_argument("--cpu-render-launcher", type=Path, default=ROOT / "scripts/run_simulator.sh")
    args = parser.parse_args()
    checks = []

    def record(name, function):
        try:
            detail = function()
            checks.append({"check": name, "ok": True, "detail": detail})
        except Exception as exc:
            checks.append({"check": name, "ok": False, "error": str(exc)})

    def imports(python, modules):
        found = probe_python(python, modules)
        failed = {name: value for name, value in found.items() if isinstance(value, dict)}
        if failed:
            raise ValueError(json.dumps(failed))
        return found

    record(
        "agent imports", lambda: imports(sys.executable, ["openai", "httpx", "dotenv", "numpy", "PIL", "openpi_client"])
    )
    record(
        "simulator imports",
        lambda: imports(
            args.sim_python,
            ["robomme.robomme_env", "robomme.env_record_wrapper", "gymnasium", "sapien", "mani_skill", "openpi_client"],
        ),
    )
    record(
        "policy imports",
        lambda: imports(
            args.policy_python,
            [
                "jax",
                "orbax.checkpoint",
                "websockets.asyncio.server",
                "openpi_client",
                "mme_vla_suite.policies.policy_config",
                "mme_vla_suite.training.config",
            ],
        ),
    )

    def credentials():
        from dotenv import dotenv_values

        values = {**dotenv_values(args.env_file), **os.environ}
        if not all(values.get(key) for key in ["BASE_URL", "API_KEY"]):
            raise ValueError("BASE_URL and API_KEY must be configured")
        if values["API_KEY"].startswith("replace-"):
            raise ValueError("API_KEY is still a placeholder")
        return "Configured; credentials are not printed and the API is not contacted"

    record("API configuration", credentials)

    def simulator():
        path = ROOT / "third_party/robomme_benchmark"
        expected = subprocess.check_output(
            ["git", "ls-tree", "HEAD", "third_party/robomme_benchmark"],
            cwd=ROOT,
            text=True,
        ).split()[2]
        actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        if actual != expected:
            raise ValueError("Simulator checkout does not match the pinned submodule")
        dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).strip()
        if dirty:
            raise ValueError("Simulator checkout has local modifications")
        for task in TASKS:
            data = json.loads(
                (path / "src/robomme/env_metadata/test" / f"record_dataset_{task}_metadata.json").read_text()
            )
            episodes = {int(row["episode"]) for row in data["records"] if row.get("seed") is not None}
            if not set(range(50)) <= episodes:
                raise ValueError(f"Missing test episode metadata: {task}")
        return "Pinned simulator and all 800 test episode seeds are available"

    record("simulator checkout and metadata", simulator)

    def checkpoint():
        path = args.checkpoint
        if path.name != "79999":
            raise ValueError("Expected the official step-79999 checkpoint")
        history = (path.parent / "history_config.txt").read_text().strip()
        if Path(history).name != "symbolic-grounded-subgoal.yaml":
            raise ValueError("Checkpoint is not configured as symbolic-grounded-subgoal")
        for relative in ["_CHECKPOINT_METADATA", "params/_METADATA", "assets/robomme/norm_stats.json"]:
            if not (path / relative).is_file():
                raise ValueError(f"Missing checkpoint file: {relative}")
        if not any(f.is_file() and f.stat().st_size for f in (path / "params").rglob("*") if f.name != "_METADATA"):
            raise ValueError("Checkpoint parameter data is missing")
        json.loads((path / "assets/robomme/norm_stats.json").read_text())
        return "GroundSG configuration, parameter files, and normalization assets are present; weights not loaded"

    record("checkpoint files", checkpoint)

    def tokenizer():
        cache = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
        path = cache / "big_vision/paligemma_tokenizer.model"
        if not path.is_file() or not path.stat().st_size:
            raise ValueError("PaliGemma tokenizer is not cached; policy startup would require a download")
        return "PaliGemma tokenizer is cached locally"

    record("tokenizer cache", tokenizer)

    initial = args.initial_experience or ROOT / "assets" / f"{args.experience_preset}-experience"
    record(
        "experience", lambda: validate_experience(initial, TASKS, initial_preset=args.experience_preset == "initial")
    )

    def launcher():
        if not args.cpu_render_launcher.is_file():
            raise ValueError("Simulator launcher does not exist")
        result = subprocess.run(["bash", "-n", str(args.cpu_render_launcher)], capture_output=True, check=False)
        if result.returncode:
            raise ValueError("Simulator launcher has invalid shell syntax")
        return "Shell syntax valid; rendering was not started"

    record("render launcher", launcher)

    def gpu():
        if shutil.which("nvidia-smi") is None:
            raise ValueError("nvidia-smi is unavailable; GPU inference readiness is unverified")
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode or not result.stdout.strip():
            raise ValueError("No accessible NVIDIA GPU; policy inference cannot be declared ready")
        return result.stdout.strip().splitlines()

    record("GPU availability", gpu)
    print(
        json.dumps(
            {
                "checks": checks,
                "not_executed": [
                    "API authentication/model request",
                    "checkpoint load/inference",
                    "renderer initialization",
                    "simulator episode",
                ],
            },
            indent=2,
        )
    )
    return int(any(not check["ok"] for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())
