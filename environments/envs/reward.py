"""
core.reward
-----------

Reusable reward helpers for the balloon environments.
"""

from __future__ import annotations
import math
import numpy as np

from environments.core.constants import ALT_MAX


# ------------------------------------------------------------------ #
# distance helpers (handle 1-, 2-, 3-D automatically)
# ------------------------------------------------------------------ #
def l2_distance(balloon_pos: np.ndarray, goal_pos: np.ndarray, dim: int) -> float:
    """
    Return Euclidean distance in the *active* dimensions:
        dim=1 → |z - z_goal|
        dim=2 → sqrt((x-xg)^2 + (y-yg)^2)
        dim=3 → full 3-D norm
    """
    if dim == 1:
        return abs(balloon_pos[-1] - goal_pos[0])
    elif dim == 2:
        return math.hypot(balloon_pos[0] - goal_pos[0],
                          balloon_pos[1] - goal_pos[1])
    else:  # dim == 3
        return float(np.linalg.norm(balloon_pos - goal_pos))


# ------------------------------------------------------------------ #
# reward functions
# ------------------------------------------------------------------ #
def distance_reward(
    balloon_pos: np.ndarray,
    goal_pos: np.ndarray,
    dim: int,
    terminated: bool,
    punishment: float,
) -> float:
    """
    Default reward: negative distance to goal, with a fixed punishment if the
    episode ended in a crash (terminated==True).
    """
    if terminated:
        return punishment

    distance = l2_distance(balloon_pos, goal_pos, dim)
    normaliser = max(ALT_MAX, 1.0)
    return -float(distance / normaliser)
