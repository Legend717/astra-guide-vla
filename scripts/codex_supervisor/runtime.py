"""Adapters for one shared simulator and a dedicated local policy worker."""

import os
import subprocess
import time
import uuid

import numpy as np

from scripts import codex_rgb_eval as rgb


class Runtime:
    def __init__(self, env):
        from robomme.env_record_wrapper import FailAwareWrapper

        self.env = env
        # Existing builder returns FailAware(Endeffector(Demonstration(env))).
        # Both routes share exactly the same DemonstrationWrapper and physical episode.
        self.joint_env = FailAwareWrapper(env.env.env)
        self.current = None
        self.prompt = None

    def reset(self):
        obs, info = self.env.reset()
        self.update(obs)
        self.prompt = str(rgb.last_scalar(info.get("task_goal"), ""))
        return obs, info

    def update(self, obs):
        if obs and len(obs.get("front_rgb_list", [])):
            self.current = {
                key: np.asarray(obs[key][-1]).copy()
                for key in ("front_rgb_list", "wrist_rgb_list", "joint_state_list", "gripper_state_list")
            }

    def image_hw(self):
        return tuple(int(x) for x in self.current["front_rgb_list"].shape[:2])

    def observation(self, subgoal):
        state = np.concatenate(
            [self.current["joint_state_list"].reshape(-1), self.current["gripper_state_list"].reshape(-1)[:1]]
        ).astype(np.float32)
        return {
            "observation/image": self.current["front_rgb_list"],
            "observation/wrist_image": self.current["wrist_rgb_list"],
            "observation/state": state,
            "prompt": self.prompt,
            "grounded_subgoal": subgoal["grounded_subgoal"],
            "simple_subgoal": subgoal["simple_subgoal"],
        }

    def step(self, action, kind):
        if kind == "joint" and self.env.unwrapped.spec.id in ("PatternLock", "RouteStick"):
            action = action[:7]  # Official stick control has no gripper actuator.
        result = (self.joint_env if kind == "joint" else self.env).step(action)
        self.update(result[0])
        return result


class WorkerPolicy:
    def __init__(self, directory, args):
        self.directory = directory / "internal/policy"
        self.directory.mkdir()
        environment = dict(os.environ)
        environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        environment.setdefault("HF_HUB_OFFLINE", "1")
        self.process = subprocess.Popen(
            [
                str(args.policy_python),
                str(rgb.ROOT / "scripts/codex_supervisor/policy.py"),
                "--directory",
                str(self.directory),
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--config",
                args.config,
                "--mode",
                args.mode,
                "--seed",
                str(args.model_seed),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
        )
        self.metadata = self.wait(self.directory / "ready.json")

    def wait(self, path):
        # No inference retry on timeout. A live worker retains ownership of the request.
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError("local policy worker exited; inspect internal failure type")
            time.sleep(0.05)
        return rgb.read_json(path)

    def call(self, operation, obs):
        identifier = uuid.uuid4().hex
        array_file = f"arrays-{identifier}.npz"
        np.savez(
            self.directory / array_file, **{key: value for key, value in obs.items() if key.startswith("observation/")}
        )
        rgb.write_json(
            self.directory / f"request-{identifier}.json",
            {
                "operation": operation,
                "arrays": array_file,
                "text": {key: value for key, value in obs.items() if not key.startswith("observation/")},
            },
        )
        result = self.wait(self.directory / f"response-{identifier}.json")
        if not result["ok"]:
            raise ValueError(
                result.get("token_budget_error", "policy rejected input or inference failed: " + result["error_type"])
            )
        return result

    def validate(self, obs):
        self.call("validate", obs)

    def infer(self, obs):
        return self.call("infer", obs)

    def close(self):
        (self.directory / "stop").touch()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
