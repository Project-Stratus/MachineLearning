"""
core.reward
-----------

Reusable reward helpers for the balloon environments.

The reward is the product of two terms:

    total = base(distance) * resource_factor(omega)

with ``total`` in [0, 1] and ``0.0`` on termination.

Distance term (``base``)
------------------------
Perciatelli-style, after Google's Balloon Learning Environment:

  - Inside ``station_radius``: flat 1.0. Once the balloon is close enough there
    is nothing to gain from chasing the exact centre, and a flat plateau avoids
    pointless oscillation about a moving optimum.
  - Outside: drops discontinuously to ``reward_dropoff`` at the boundary (the
    "cliff" — crossing out of the station is immediately expensive), then
    decays exponentially with ``reward_halflife`` metres.
  - Terminated: 0.0. Dying forfeits the whole remaining reward stream, which is
    a stronger and better-scaled punishment than any fixed negative constant.

Resource term (``resource_factor``)
-----------------------------------
``omega`` = fraction of the *initial* consumable budget spent so far
(ZP: ballast dropped + helium vented; SP: air pumped, as a proxy). Then::

    f = 1.0                                              if omega == 0
    f = max(0, RESOURCE_PENALTY_BASE
                - RESOURCE_PENALTY_SLOPE * omega)        otherwise

Ballast and helium are irreversible and the budget binds, but without this term
the only cost of spending them is the eventual terminal death — nothing stops
the agent emptying its budget in the first hour.

Why multiplicative and not additive
-----------------------------------
An additive penalty (``base - c * omega``) has a magnitude that is independent
of where the balloon is, so far from the station — exactly where ``base`` is
small — the penalty term dominates the action-value estimates. The agent then
learns "never actuate" rather than "actuate only when it buys station time",
and the distance gradient it is supposed to be following is drowned out. This
was Loon's finding for its power budget (a ``x(0.95 - 0.3 * omega)`` factor beat
the additive form); our ballast/helium budget is likewise binding and
irreversible, so the same reasoning transfers even though the resource does
not. A multiplicative factor instead scales the whole return uniformly: it
never changes the *sign* of the distance gradient, it just makes every future
reward worth proportionally less once the budget has been eaten into.

Note the deliberate discontinuity at ``omega == 0``: the first unit of
consumption costs ``1 - RESOURCE_PENALTY_BASE`` outright. This is Loon's design
and it prices "touching the actuators at all" separately from "how much".
"""

from __future__ import annotations
import math
import numpy as np

from environments.core.constants import (
    STATION_RADIUS, REWARD_DROPOFF, REWARD_HALFLIFE,
    RESOURCE_PENALTY_BASE, RESOURCE_PENALTY_SLOPE,
)


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


def _resource_factor(
    resource_consumed_frac: float,
    base: float = RESOURCE_PENALTY_BASE,
    slope: float = RESOURCE_PENALTY_SLOPE,
) -> float:
    """Multiplicative resource-consumption factor in [0, 1].

    Returns exactly 1.0 while nothing has been consumed, otherwise
    ``base - slope * omega`` floored at 0.0. ``omega`` is clipped into
    [0, 1] first, so an over-budget accounting error cannot push the
    reward above 1.0 or make the factor grow without bound.
    """
    omega = float(resource_consumed_frac)
    if not math.isfinite(omega) or omega <= 0.0:
        return 1.0
    omega = min(omega, 1.0)
    return max(0.0, base - slope * omega)


def balloon_reward(
    *,
    balloon_pos: np.ndarray,
    goal_pos: np.ndarray,
    dim: int,
    terminated: bool,
    resource_consumed_frac: float = 0.0,
    station_radius: float = STATION_RADIUS,
    reward_dropoff: float = REWARD_DROPOFF,
    reward_halflife: float = REWARD_HALFLIFE,
) -> tuple[float, dict[str, float], float]:
    """Composite reward used by the Balloon environments.

    Distance shaping (Perciatelli-style):
      - Inside *station_radius*: flat reward of 1.0
      - Outside: drops to *reward_dropoff* at the boundary, then
        exponentially decays with *reward_halflife* (in metres).
      - On termination: reward is 0.0 (losing all future reward is
        the punishment).

    Resource rationing:
      *resource_consumed_frac* (omega) is the fraction of the initial
      consumable budget spent so far, in [0, 1]. It scales the distance
      reward **multiplicatively** — see the module docstring for why an
      additive penalty is the wrong shape here.

    Parameters
    ----------
    balloon_pos, goal_pos, dim
        Passed straight to :func:`l2_distance`.
    terminated
        If True the reward is 0.0 and every component is 0.0.
    resource_consumed_frac
        omega in [0, 1]. Values <= 0 (or non-finite) mean "nothing spent"
        and give a factor of exactly 1.0; values above 1 are clipped.

    Returns
    -------
    (total, components, distance)
        ``components`` keys are exactly ``station``, ``decay``, ``base``
        and ``resource_factor``, with
        ``base == station + decay`` and ``total == base * resource_factor``.
    """
    distance = l2_distance(balloon_pos, goal_pos, dim)

    if terminated:
        components = dict(station=0.0, decay=0.0, base=0.0, resource_factor=0.0)
        return 0.0, components, distance

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

    base = station_component + decay_component
    resource_factor = _resource_factor(resource_consumed_frac)
    total = base * resource_factor

    components = dict(
        station=station_component,
        decay=decay_component,
        base=base,
        resource_factor=resource_factor,
    )
    return total, components, distance


# ------------------------------------------------------------------ #
# evaluation metrics
# ------------------------------------------------------------------ #
def time_within_radius(
    distances, station_radius: float = STATION_RADIUS
) -> float:
    """Time-within-radius (TWR): fraction of samples inside *station_radius*.

    Parameters
    ----------
    distances
        Sequence or ndarray of distances to the goal, one per sample
        (typically one per decision step). Nested arrays are flattened,
        so a stack of per-episode distance traces of equal length is
        accepted and pooled.
    station_radius
        Inclusive boundary — a sample exactly at the radius counts as
        inside, matching :func:`balloon_reward`.

    Returns
    -------
    float
        Fraction in [0, 1]. **Empty input returns 0.0**, not NaN: TWR is
        logged and averaged across evaluation episodes, and a NaN would
        poison every downstream aggregate. "No samples observed" and
        "no samples on station" are treated alike — pessimistically —
        which is the safe direction for a metric used to pick the best
        checkpoint.
    """
    arr = np.asarray(distances, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(arr <= station_radius) / arr.size)
