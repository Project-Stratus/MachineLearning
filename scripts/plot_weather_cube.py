#!/usr/bin/env python3
"""Plot an ERA5 runtime cube and optional 12-hour baseline trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agents.baselines import GreedyWindAgent, PassiveDriftAgent  # noqa: E402
from environments.core.constants import DECISION_INTERVAL, TIME_MAX  # noqa: E402
from environments.core.weather import WeatherCube  # noqa: E402
from environments.envs.balloon_3d_env import Balloon3DEnv  # noqa: E402
from environments.wrappers.decision_interval import (  # noqa: E402
    DecisionIntervalWrapper,
)


def rollout_baselines(cube_path: Path, seed: int) -> dict[str, np.ndarray]:
    """Run passive and greedy policies against one cube."""

    cube = WeatherCube.load(cube_path)
    time_max = min(TIME_MAX, int(cube.time_s[-1] - cube.time_s[0]))
    if time_max < DECISION_INTERVAL:
        raise ValueError("weather cube is too short for a baseline trajectory")

    trajectories: dict[str, np.ndarray] = {}
    policies = {
        "Passive drift": PassiveDriftAgent(),
        "Greedy wind": GreedyWindAgent(),
    }
    for name, policy in policies.items():
        env = DecisionIntervalWrapper(
            Balloon3DEnv(
                dim=3,
                config={"weather_path": str(cube_path), "time_max": time_max},
            ),
            decision_interval=DECISION_INTERVAL,
        )
        observation, _ = env.reset(seed=seed)
        positions = [env.unwrapped._balloon.pos.copy()]
        while True:
            action, _ = policy.predict(observation, deterministic=True)
            observation, _reward, terminated, truncated, _info = env.step(int(action))
            positions.append(env.unwrapped._balloon.pos.copy())
            if terminated or truncated:
                break
        trajectories[name] = np.asarray(positions)
        env.close()
    return trajectories


def plot_cube(
    cube_path: Path,
    output: Path,
    *,
    altitude_m: float = 20_000.0,
    seed: int = 1_234_567,
    include_trajectories: bool = True,
) -> None:
    """Write a horizontal field and London time/height section."""

    cube = WeatherCube.load(cube_path)
    trajectories = rollout_baselines(cube_path, seed) if include_trajectories else {}
    altitude_index = int(np.argmin(np.abs(cube.altitude_m - altitude_m)))
    x_km = cube.x_m / 1_000.0
    y_km = cube.y_m / 1_000.0
    u = cube.u[0, altitude_index]
    v = cube.v[0, altitude_index]
    speed = np.hypot(u, v)
    colours = {"Passive drift": "tab:red", "Greedy wind": "tab:orange"}

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    horizontal = axes[0].pcolormesh(x_km, y_km, speed, shading="auto", cmap="viridis")
    stride = max(1, min(cube.x_m.size, cube.y_m.size) // 14)
    axes[0].quiver(
        x_km[::stride],
        y_km[::stride],
        u[::stride, ::stride],
        v[::stride, ::stride],
        color="white",
        alpha=0.75,
        scale=220,
        width=0.0025,
    )
    for name, positions in trajectories.items():
        axes[0].plot(
            positions[:, 0] / 1_000.0,
            positions[:, 1] / 1_000.0,
            color=colours[name],
            linewidth=2.0,
            label=name,
        )
        axes[0].scatter(
            positions[0, 0] / 1_000.0,
            positions[0, 1] / 1_000.0,
            color=colours[name],
            marker="o",
            s=35,
        )
        axes[0].scatter(
            positions[-1, 0] / 1_000.0,
            positions[-1, 1] / 1_000.0,
            color=colours[name],
            marker="x",
            s=55,
        )
    axes[0].scatter(
        0.0,
        0.0,
        color="white",
        edgecolor="black",
        marker="*",
        s=140,
        label="London",
    )
    axes[0].set_title(
        f"Horizontal wind at {cube.altitude_m[altitude_index] / 1_000.0:.1f} km"
        "\ncircle=start, ×=finish"
    )
    axes[0].set_xlabel("East of London (km)")
    axes[0].set_ylabel("North of London (km)")
    axes[0].set_aspect("equal")
    axes[0].legend(loc="lower left")
    figure.colorbar(horizontal, ax=axes[0], label="Wind speed (m/s)")

    center_y = int(np.argmin(np.abs(cube.y_m)))
    center_x = int(np.argmin(np.abs(cube.x_m)))
    band = (cube.altitude_m >= 15_000.0) & (cube.altitude_m <= 25_000.0)
    column_speed = np.hypot(
        cube.u[:, band, center_y, center_x],
        cube.v[:, band, center_y, center_x],
    ).T
    time_height = axes[1].pcolormesh(
        cube.time_s / 3_600.0,
        cube.altitude_m[band] / 1_000.0,
        column_speed,
        shading="auto",
        cmap="magma",
    )
    for name, positions in trajectories.items():
        elapsed_hours = np.linspace(
            cube.time_s[0] / 3_600.0,
            cube.time_s[-1] / 3_600.0,
            positions.shape[0],
        )
        axes[1].plot(
            elapsed_hours,
            positions[:, 2] / 1_000.0,
            color=colours[name],
            linewidth=2.0,
            label=name,
        )
    axes[1].set_title("Wind-speed profile above London\nand trajectory altitude")
    axes[1].set_xlabel("Hours from cube start")
    axes[1].set_ylabel("Altitude (km)")
    axes[1].set_ylim(15.0, 25.0)
    if trajectories:
        axes[1].legend(loc="lower right")
    figure.colorbar(time_height, ax=axes[1], label="Wind speed (m/s)")

    start = cube.metadata.get("start_time_utc", "unknown start")
    figure.suptitle(f"{cube.metadata.get('site_name', 'ERA5')} weather — {start}")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--altitude", type=float, default=20_000.0)
    parser.add_argument("--seed", type=int, default=1_234_567)
    parser.add_argument(
        "--no-trajectories",
        action="store_true",
        help="Plot weather only without running passive/greedy policies.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_cube(
        args.input,
        args.output,
        altitude_m=args.altitude,
        seed=args.seed,
        include_trajectories=not args.no_trajectories,
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
