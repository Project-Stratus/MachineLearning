"""The onboarding toy task: hold a balloon at a target altitude.

The balloon starts somewhere within 1.5 km of a target altitude. It has to get
into a +/-250 m band around the target and stay there for a 2-hour flight.
Every 3 minutes the agent picks one of three actions:

    0  vent helium    -> the balloon gets heavier and sinks
    1  do nothing
    2  drop ballast   -> the balloon gets lighter and rises

Both actions are irreversible, and there is only enough ballast for 20 drops.
If the balloon strays more than 3 km from the target it is lost and the
episode ends early.

The physics is the project's own zero-pressure balloon model
(``environments.core.balloon.Balloon``), stepped every 5 seconds.

Notebook 2 builds this environment step by step. This file is the finished
version, which notebooks 3 and 4 import so that each notebook stands alone,
plus the small running, scoring and plotting helpers the notebooks share.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

from environments.core.balloon import Balloon
from environments.core.constants import BALLAST_DROP

# --------------------------------------------------------------------------- #
# The task
# --------------------------------------------------------------------------- #
TARGET_ALTITUDE = 20_000.0  # m
BAND = 250.0  # m either side of the target counts as "on station"
START_SPREAD = 1_500.0  # m: the start is uniform in target +/- this
LOST_DISTANCE = 3_000.0  # m: further than this from the target ends the episode

DECISION_SECONDS = 180  # one decision every 3 minutes
PHYSICS_DT = 5.0  # s per physics step, so 36 physics steps per decision
FLIGHT_DECISIONS = 40  # 40 x 3 min = a 2-hour flight

BALLAST_DROPS = 20  # ballast on board, counted in drops

VENT, STAY, DROP = 0, 1, 2
ACTION_NAMES = ("vent", "stay", "drop")

# Observations are scaled to roughly [-1, 1] so a neural network can use them
# directly. Multiply back by these to get metres and metres per second.
ALTITUDE_SCALE = LOST_DISTANCE  # m
VELOCITY_SCALE = 2.0  # m/s
OBS_NAMES = ("altitude_error", "velocity", "ballast_left")

#: Seeds held back for evaluation. Training should use different seeds, so a
#: score on these says how well a policy copes with flights it has never seen.
EVAL_SEEDS = range(10_000, 10_050)


def default_reward(altitude_error: float, action: int) -> float:
    """1 per decision inside the band, decaying outside it, minus 0.1 per action.

    The reward halves for every 250 m outside the band.
    """
    distance = abs(altitude_error)
    if distance <= BAND:
        reward = 1.0
    else:
        reward = 0.5 ** ((distance - BAND) / 250.0)
    if action != STAY:
        reward -= 0.1
    return reward


class AltitudeHoldEnv(gym.Env):
    """Gymnasium environment for the altitude-hold task.

    Parameters
    ----------
    reward_fn : callable, optional
        ``reward_fn(altitude_error, action) -> float``, where
        ``altitude_error`` is in metres (positive means too high). Defaults to
        :func:`default_reward`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, reward_fn: Callable[[float, int], float] = default_reward
    ) -> None:
        self.reward_fn = reward_fn
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.balloon: Balloon | None = None
        self.decisions = 0

    def _get_obs(self) -> np.ndarray:
        b = self.balloon
        return np.array(
            [
                np.clip((b.altitude - TARGET_ALTITUDE) / ALTITUDE_SCALE, -1.0, 1.0),
                np.clip(b.velocity / VELOCITY_SCALE, -1.0, 1.0),
                b.ballast_mass / (BALLAST_DROPS * BALLAST_DROP),
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> dict:
        return {
            "altitude": float(self.balloon.altitude),
            "velocity": float(self.balloon.velocity),
            "ballast_drops_left": round(self.balloon.ballast_mass / BALLAST_DROP),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)  # seeds self.np_random
        start = TARGET_ALTITUDE + self.np_random.uniform(-START_SPREAD, START_SPREAD)
        # Balloon() computes how much helium makes it float at `start`, so the
        # balloon begins at rest.
        self.balloon = Balloon(
            dim=1,
            ballast_initial=BALLAST_DROPS * BALLAST_DROP,
            position=np.array([start]),
        )
        self.decisions = 0
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        b = self.balloon
        if action == DROP and not b.is_ballast_empty:
            b.drop_ballast()
        elif action == VENT:
            b.vent_gas()

        for _ in range(int(DECISION_SECONDS / PHYSICS_DT)):
            b.update(PHYSICS_DT)
        self.decisions += 1

        altitude_error = b.altitude - TARGET_ALTITUDE
        # bool(): Gymnasium wants Python bools, and numpy comparisons give
        # numpy bools. check_env() catches this.
        terminated = bool(abs(altitude_error) > LOST_DISTANCE)
        truncated = bool(self.decisions >= FLIGHT_DECISIONS)
        # Once the balloon is lost it earns nothing more: losing all the
        # reward it would have collected is the penalty.
        reward = 0.0 if terminated else float(self.reward_fn(altitude_error, action))
        return self._get_obs(), reward, terminated, truncated, self._get_info()


# --------------------------------------------------------------------------- #
# A hand-written controller to compare learned agents against
# --------------------------------------------------------------------------- #
def hold_controller(obs: np.ndarray) -> int:
    """Head for the target at up to 1 m/s, aiming to arrive in about 10 minutes.

    This is notebook 2's answer. It is the bar a learned agent has to clear.
    """
    altitude_error = obs[0] * ALTITUDE_SCALE
    velocity = obs[1] * VELOCITY_SCALE
    wanted_velocity = float(np.clip(-altitude_error / 600.0, -1.0, 1.0))
    if velocity < wanted_velocity - 0.25:
        return DROP
    if velocity > wanted_velocity + 0.25:
        return VENT
    return STAY


# --------------------------------------------------------------------------- #
# Running, scoring and plotting
# --------------------------------------------------------------------------- #
def run_episode(env: gym.Env, policy: Callable[[np.ndarray], int], seed: int) -> dict:
    """Fly one episode and record what happened.

    ``policy(obs) -> action``. Returns a dict of per-decision arrays
    (``minutes``, ``altitude``, ``action``, ``reward``) plus the episode
    totals ``total_reward`` and ``lost``.
    """
    obs, info = env.reset(seed=seed)
    minutes, altitude, actions, rewards = [0.0], [info["altitude"]], [], []
    done = False
    while not done:
        action = int(policy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        minutes.append(minutes[-1] + DECISION_SECONDS / 60)
        altitude.append(info["altitude"])
        actions.append(action)
        rewards.append(reward)
        done = terminated or truncated
    return {
        "minutes": np.array(minutes),
        "altitude": np.array(altitude),
        "action": np.array(actions),
        "reward": np.array(rewards),
        "total_reward": float(np.sum(rewards)),
        "lost": bool(terminated),
    }


def evaluate(
    policy: Callable[[np.ndarray], int],
    env: gym.Env | None = None,
    seeds=EVAL_SEEDS,
) -> dict:
    """Average a policy's performance over the evaluation flights.

    Returns the mean ``total_reward``, the fraction of time spent ``in_band``,
    the mean number of ``actions`` taken (vents + drops) and the fraction of
    flights ``lost``.
    """
    env = env if env is not None else AltitudeHoldEnv()
    flights = [run_episode(env, policy, seed) for seed in seeds]
    in_band = [
        np.mean(np.abs(f["altitude"][1:] - TARGET_ALTITUDE) <= BAND) for f in flights
    ]
    return {
        "total_reward": float(np.mean([f["total_reward"] for f in flights])),
        "in_band": float(np.mean(in_band)),
        "actions": float(np.mean([np.sum(f["action"] != STAY) for f in flights])),
        "lost": float(np.mean([f["lost"] for f in flights])),
    }


def check(passed: bool, message: str) -> None:
    """Print a tick or a cross for a notebook self-check."""
    print(("✅ " if passed else "❌ ") + message)


def plot_flights(flights, labels=None, ax=None, title=None):
    """Altitude against time for one or more flights, with the band shaded."""
    if isinstance(flights, dict):
        flights = [flights]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    ax.axhspan(
        (TARGET_ALTITUDE - BAND) / 1000,
        (TARGET_ALTITUDE + BAND) / 1000,
        color="tab:green",
        alpha=0.15,
        label="target band",
    )
    for i, flight in enumerate(flights):
        label = labels[i] if labels is not None else None
        ax.plot(flight["minutes"], flight["altitude"] / 1000, label=label)
    ax.set_xlabel("time (minutes)")
    ax.set_ylabel("altitude (km)")
    ax.set_xlim(0, FLIGHT_DECISIONS * DECISION_SECONDS / 60)
    if title:
        ax.set_title(title)
    if labels is not None:
        ax.legend(loc="best", fontsize="small")
    return ax


def plot_learning_curve(returns, label=None, ax=None, window=50):
    """Training return per episode, smoothed over ``window`` episodes."""
    returns = np.asarray(returns, dtype=float)
    if len(returns) < window:
        raise ValueError(
            f"Need at least {window} episodes to plot a learning curve, got "
            f"{len(returns)}. Has the agent been trained yet?"
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))
    smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
    ax.plot(np.arange(window, len(returns) + 1), smoothed, label=label)
    ax.set_xlabel("training episode")
    ax.set_ylabel(f"return (mean of last {window})")
    return ax
