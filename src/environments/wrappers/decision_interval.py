"""
wrappers.decision_interval
--------------------------

Gym wrapper that decouples the agent's decision frequency from the
physics timestep.  The agent chooses an action once per *decision
interval* (default 60 physics steps = 1 minute at DT=1 s).

Action semantics
~~~~~~~~~~~~~~~~
The chosen action (vent / nothing / drop ballast) fires **once** on the
first sub-step.  The remaining sub-steps execute with action index 1
("nothing").  This matches the real hardware: one valve opening per
decision, then the balloon drifts until the next decision.

Reward
~~~~~~
Per-step rewards are **summed** over the interval.  Each second spent
in the station-keeping zone contributes +1.0, so the maximum reward
per decision is equal to the interval length.

Termination
~~~~~~~~~~~
If the balloon terminates mid-interval (crash, pop, deflate, ballast
exhausted, out of bounds), the wrapper returns immediately with the
accumulated reward up to that point and ``terminated=True``.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from environments.core.constants import DECISION_INTERVAL


class DecisionIntervalWrapper(gym.Wrapper):
    """Step the physics *decision_interval* times per agent action."""

    # Action index for "do nothing" in the base env's Discrete(3) space
    _NOOP = 1

    def __init__(self, env: gym.Env, decision_interval: int = DECISION_INTERVAL):
        super().__init__(env)
        self.decision_interval = decision_interval

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for i in range(self.decision_interval):
            # Fire the real action on the first sub-step only
            sub_action = action if i == 0 else self._NOOP
            obs, reward, terminated, truncated, info = self.env.step(sub_action)
            total_reward += reward
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
