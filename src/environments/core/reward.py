"""
core.reward
-----------

Reusable reward helpers for the balloon environments.

Reward style inspired by Google's Balloon Learning Environment (Perciatelli):
  - Inside station-keeping radius: flat 1.0
  - Outside radius: exponential decay with configurable half-life
  - Terminated: 0.0 (no extra punishment — dying forfeits future reward)
  - Reward range: [0, 1]
"""

from __future__ import annotations
import math
import numpy as np


# ------------------------------------------------------------------ #
# distance helpers (handle 1-, 2-, 3-D automatically)
# ------------------------------------------------------------------ #
def l2_distance(balloon_pos: np.ndarray, goal_pos: np.ndarray, dim: int) -> float:
    """
    Return Euclidean distance in the *reward-relevant* dimensions:
        dim=1 → |z - z_goal|  (altitude only)
        dim=2 → sqrt((x-xg)^2 + (y-yg)^2)
        dim=3 → sqrt((x-xg)^2 + (y-yg)^2)  (x-y only; altitude is the
                 agent's control mechanism, not an objective)
    """
    if dim == 1:
        return abs(float(balloon_pos[-1]) - float(goal_pos[0]))
    else:  # dim 2 and 3: horizontal distance only
        dx = float(balloon_pos[0]) - float(goal_pos[0])
        dy = float(balloon_pos[1]) - float(goal_pos[1])
        return math.sqrt(dx * dx + dy * dy)


# ------------------------------------------------------------------ #
# reward functions
# ------------------------------------------------------------------ #
_LN_HALF = -0.69314718056  # ln(0.5), used for exponential decay


def balloon_reward(
    *,
    balloon_pos: np.ndarray,
    goal_pos: np.ndarray,
    dim: int,
    terminated: bool,
    station_radius: float = 10_000.0,
    reward_dropoff: float = 0.4,
    reward_halflife: float = 20_000.0,
) -> tuple[float, dict[str, float], float]:
    """Composite reward used by the Balloon environments.

    Inspired by Google BLE's Perciatelli reward:
      - Inside *station_radius*: flat reward of 1.0
      - Outside: drops to *reward_dropoff* at the boundary, then
        exponentially decays with *reward_halflife* (in metres).
      - On termination: reward is 0.0 (losing all future reward is
        the punishment).

    Returns the total reward, a component breakdown, and the current
    distance to goal.
    """
    distance = l2_distance(balloon_pos, goal_pos, dim)

    if terminated:
        total = 0.0
        components = dict(station=0.0, decay=0.0)
        return total, components, distance

    if distance <= station_radius:
        station_component = 1.0
        decay_component = 0.0
    else:
        station_component = 0.0
        excess = distance - station_radius
        halflife_m = max(reward_halflife, 1.0)
        decay_component = reward_dropoff * math.exp(
            _LN_HALF / halflife_m * excess
        )

    total = station_component + decay_component
    components = dict(
        station=station_component,
        decay=decay_component,
    )
    return total, components, distance
