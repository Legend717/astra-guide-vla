"""Local VLA worker: exact training config, post-transform actions, strict token limits.

Runs in the policy Python environment. No network model API or runtime VLM.
The simulator exchanges only allowed observation arrays through internal files.
"""

import argparse
import dataclasses
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import codex_rgb_eval as rgb  # noqa: E402


class LocalPolicy:
    def __init__(self, checkpoint, config_name, mode, seed):
        from mme_vla_suite.policies import policy_config
        from mme_vla_suite.training import config

        expected = "simple_subgoal" if mode == "simple-smoke" else "grounded_subgoal"
        # create_trained_policy resolves this file itself; reject missing or mismatching metadata first.
        history = (checkpoint.parent / "history_config.txt").read_text().strip()
        loaded = config.get_history_config(history)
        if loaded.representation_type != "symbolic" or loaded.symbolic_memory.type != expected:
            raise ValueError("checkpoint history config does not match the requested symbolic mode")
        training = config.get_config(config_name)
        training = dataclasses.replace(
            training, model=dataclasses.replace(training.model, history_config=history, use_history=True)
        )
        self.policy = policy_config.create_trained_policy(training, checkpoint, seed=seed, reject_token_truncation=True)
        self.key = expected
        if self.policy.config.representation_type != "symbolic" or self.policy.config.symbolic_memory.type != expected:
            raise ValueError("loaded model type mismatch")
        self.metadata = {
            "checkpoint": str(checkpoint.resolve()),
            "training_config": config_name,
            "actual_model_type": expected,
            "model_class": type(self.policy._model).__name__,  # noqa: SLF001 - audit actual loaded model
            "backbone_model_type": str(training.model.model_type),
            "history_config": history,
            "mode": mode,
            "seed": seed,
            "token_budget_policy": "reject truncation in both actual prompt tokenization paths",
            "actions": "post output transforms: 7 absolute joint radians + gripper command",
        }

    def validate(self, obs):
        # The real transforms include normalization and both complete tokenized texts.
        self.policy._input_transform(dict(obs))  # noqa: SLF001 - validate the exact trained transform pipeline

    def infer(self, obs):
        self.validate(obs)
        return self.policy.infer(obs)


def worker(args):
    import numpy as np

    directory = args.directory.resolve()
    with rgb.quiet_simulator():
        policy = LocalPolicy(args.checkpoint.resolve(), args.config, args.mode, args.seed)
    rgb.write_json(directory / "ready.json", {"pid": os.getpid(), **policy.metadata})
    while not (directory / "stop").exists():
        for path in sorted(directory.glob("request-*.json")):
            output = directory / path.name.replace("request-", "response-")
            if output.exists():
                continue
            request = rgb.read_json(path)
            try:
                with np.load(directory / request["arrays"], allow_pickle=False) as data:
                    obs = {key: data[key] for key in data.files}
                obs.update(request["text"])
                with rgb.quiet_simulator():
                    if request["operation"] == "validate":
                        policy.validate(obs)
                        result = {}
                    else:
                        result = {"actions": np.asarray(policy.infer(obs)["actions"]).tolist()}
                rgb.write_json(output, {"ok": True, **result})
            except Exception as exc:
                # Never expose model diagnostics or arbitrary exception content.
                result = {"ok": False, "error_type": type(exc).__name__}
                if isinstance(exc, ValueError) and re.fullmatch(
                    r"Full model text requires \d+ tokens; limit is \d+", str(exc)
                ):
                    result["token_budget_error"] = str(exc)
                rgb.write_json(output, result)
        time.sleep(0.03)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="mme_vla_suite")
    parser.add_argument("--mode", choices=("grounded", "simple-smoke"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        worker(args)
    except Exception as exc:
        rgb.write_json(args.directory / "failed.json", {"error_type": type(exc).__name__})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
