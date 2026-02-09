"""
core.reward
-----------

Reusable reward helpers for the balloon environments.
"""

from __future__ import annotations
import math
import numpy as np

def _normalise_distance(distance: float, max_distance: float) -> float:
    """Return a distance scaled to [-1, 0], clamped to avoid division by zero."""
    normaliser = max(max_distance, 1.0)
    return -float(distance / normaliser)


# ------------------------------------------------------------------ #
# distance helpers (handle 1-, 2-, 3-D automatically)
# ------------------------------------------------------------------ #
def l2_distance(balloon_pos: np.ndarray, goal_pos: np.ndarray, dim: int) -> float:
    """
    Return Euclidean distance in the *active* dimensions:
        dim=1 → |z - z_goal|  (balloon uses last index for altitude)
        dim=2 → sqrt((x-xg)^2 + (y-yg)^2)
        dim=3 → full 3-D norm
    """
    if dim == 1:
        return abs(float(balloon_pos[-1]) - float(goal_pos[0]))
    elif dim == 2:
        dx = float(balloon_pos[0]) - float(goal_pos[0])
        dy = float(balloon_pos[1]) - float(goal_pos[1])
        return math.sqrt(dx * dx + dy * dy)
    else:
        dx = float(balloon_pos[0]) - float(goal_pos[0])
        dy = float(balloon_pos[1]) - float(goal_pos[1])
        dz = float(balloon_pos[2]) - float(goal_pos[2])
        return math.sqrt(dx * dx + dy * dy + dz * dz)


# ------------------------------------------------------------------ #
# reward functions
# ------------------------------------------------------------------ #
def distance_reward(
    balloon_pos: np.ndarray,
    goal_pos: np.ndarray,
    dim: int,
    terminated: bool,
    punishment: float,
    max_distance: float,
) -> float:
    """Negative normalised distance to goal with a crash punishment."""
    if terminated:
        return punishment

    distance = l2_distance(balloon_pos, goal_pos, dim)
    return _normalise_distance(distance, max_distance)


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
    max_distance: float,
    success_radius: float = 500.0,        # inner radius for full bonus (m)
    success_outer_radius: float = 1500.0,  # outer radius where ramp begins (m)
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
        components = dict(distance=punishment, direction=0.0, reached=0.0, effect=0.0)
        return total, components, distance

    # DISTANCE reward
    # Normalised distance to goal, in [-1, 0]
    distance_component = _normalise_distance(distance, max_distance)

    # DIRECTION reward
    # Positive when distance is shrinking, negative when growing
    # Scaled to [-1, 1] based on success_radius
    distance_delta = prev_distance - distance
    if success_radius <= 0.0:
        scaled_delta = 0.0
    else:
        scaled_delta = distance_delta / success_radius
        if scaled_delta > 1.0:
            scaled_delta = 1.0
        elif scaled_delta < -1.0:
            scaled_delta = -1.0
    direction_component = direction_scale * scaled_delta

    # REACHED reward
    # Graduated bonus: linear ramp from success_outer_radius to
    # success_radius, full bonus inside success_radius when slow.
    speed2 = 0.0
    for i in range(dim):
        v = float(velocity[i])
        speed2 += v * v
    reached_component = 0.0
    if distance < success_outer_radius:
        proximity = (success_outer_radius - distance) / (success_outer_radius - success_radius)
        if proximity > 1.0:
            proximity = 1.0
        reached_component = 0.3 * proximity
        # Extra: full bonus only when inside inner radius AND moving slowly
        if distance < success_radius and speed2 < success_speed * success_speed:
            reached_component = 0.3

    effect_component = 0.0

    total = distance_component + direction_component + reached_component + effect_component
    components = dict(
        distance=distance_component,
        direction=direction_component,
        reached=reached_component,
        effect=effect_component,
    )
    return total, components, distance
