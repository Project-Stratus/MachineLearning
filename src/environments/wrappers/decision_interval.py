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

Resource penalties
~~~~~~~~~~~~~~~~~~
Deliberately *not* handled here.  Because the real action fires on one
sub-step only, charging its resource penalty on that sub-step alone would
dilute it across the interval — a 3% penalty landing on 1 of 60 summed
rewards is a 0.05% penalty, which shapes nothing.  The environment
therefore holds the penalty for a full decision interval itself
(``Balloon3DEnv._charge_resources``), so it is priced correctly with or
without this wrapper.  All this wrapper does is tell the env what interval
it is running at, via ``set_decision_interval``.

Termination
~~~~~~~~~~~
If the balloon terminates mid-interval (deflated, ballast exhausted, or
the numerical abort), the wrapper returns immediately with the accumulated
reward up to that point and ``terminated=True``.  Altitude limits no longer
terminate — the env's safety layer clamps instead.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym

from environments.core.constants import DECISION_INTERVAL


class DecisionIntervalWrapper(gym.Wrapper):
    """Step the physics *decision_interval* times per agent action."""

    # Action index for "do nothing" in the base env's Discrete(3) space
    _NOOP = 1

    def __init__(self, env: gym.Env, decision_interval: int | None = None):
        super().__init__(env)

        base = env.unwrapped
        if decision_interval is None:
            cfg = getattr(base, "cfg", None)
            decision_interval = (
                cfg.get("decision_interval", DECISION_INTERVAL)
                if isinstance(cfg, dict) else DECISION_INTERVAL
            )
        self.decision_interval = max(1, int(decision_interval))

        # One source of truth: the env's resource-penalty hold must span the
        # same number of physics steps this wrapper does.
        setter = getattr(base, "set_decision_interval", None)
        if callable(setter):
            setter(self.decision_interval)

    def step(self, action: Any):
        total_reward = 0.0
        terminated = False
        truncated = False
        obs = None
        info: dict[str, Any] = {}

        for i in range(self.decision_interval):
            # Fire the real action on the first sub-step only
            sub_action = action if i == 0 else self._NOOP
            obs, reward, terminated, truncated, info = self.env.step(sub_action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info
