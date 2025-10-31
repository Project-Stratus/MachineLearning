"""
core.reward
-----------

Reusable reward helpers for the balloon environments.
"""

from __future__ import annotations
import math
import numpy as np

from environments.core.constants import ALT_MAX


def _normalise_distance(distance: float) -> float:
    """Return a distance scaled to [-1, 0], clamped to avoid division by zero."""
    normaliser = max(ALT_MAX, 1.0)
    return -float(distance / normaliser)


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
    """Negative normalised distance to goal with a crash punishment."""
    if terminated:
        return punishment

    distance = l2_distance(balloon_pos, goal_pos, dim)
    return _normalise_distance(distance)


def balloon_reward(
    *,
    balloon_pos: np.ndarray,
    goal_pos: np.ndarray,
    velocity: np.ndarray,
    dim: int,
    terminated: bool,
    effect: int,
    punishment: float,
    prev_distance: float,
    success_radius: float = 150.0,      # radius (m)
    success_speed: float = 0.2,
    direction_scale: float = 0.05,
) -> tuple[float, dict[str, float], float]:
    """Composite reward used by the Balloon environments.

    Returns the total reward, a component breakdown, and the updated
    reference distance for the next step.
    """

    distance = l2_distance(balloon_pos, goal_pos, dim)

    if terminated:
        total = punishment
        components = dict(distance=punishment, direction=0.0, reached=0.0)
        return total, components, distance

    # DISTANCE reward
    # Normalised distance to goal, in [-1, 0]
    distance_component = _normalise_distance(distance)

    # DIRECTION reward
    # Positive when distance is shrinking, negative when growing
    # Scaled to [-1, 1] based on success_radius
    distance_delta = float(prev_distance - distance)
    if success_radius <= 0.0:
        scaled_delta = 0.0
    else:
        scaled_delta = np.clip(distance_delta / success_radius, -1.0, 1.0)
    direction_component = direction_scale * scaled_delta

    # REACHED reward
    # Bonus when near the target and moving slowly
    speed = float(np.linalg.norm(velocity[:dim]))
    reached = float(distance < success_radius and speed < success_speed)
    reached_component = 0.3 * reached

    # DO NOTHING reward
    # Small bonus for taking no action (effect=0)
    # Likely to be replaced by some sort of 'energy' measure
    # if effect == 0:
    #     effect_component = 0.001
    # else:
    #     effect_component = 0.0
    effect_component = 0.0

    total = distance_component + direction_component + reached_component + effect_component
    components = dict(
        distance=distance_component,
        direction=direction_component,
        reached=reached_component,
        effect=effect_component,
    )
    return total, components, distance
