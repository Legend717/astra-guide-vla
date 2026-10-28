"""Explicit engineering backend: hold measured arm joints, without loading a VLA."""

from typing import ClassVar

import numpy as np


class StallPolicy:
    metadata: ClassVar[dict] = {
        "backend": "mock-stall",
        "real_vla": False,
        "description": "Repeats measured joints with an open gripper; no task understanding or checkpoint.",
    }

    def validate(self, obs):
        state = np.asarray(obs["observation/state"])
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("mock policy needs eight finite robot state values")

    def infer(self, obs):
        self.validate(obs)
        action = np.asarray(obs["observation/state"], dtype=float).copy()
        action[7] = 1.0
        return {"actions": np.tile(action, (32, 1)).tolist()}

    def close(self):
        pass
