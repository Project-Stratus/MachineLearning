"""Layer 1 demo: watch the trained agent station-keep. No arguments, just run it.

    python demo.py

Plays the best held-out scenarios back to back with the pygame renderer, then
prints the time-within-radius for each alongside the baselines it beat.

Two things this does that ``python main.py --dim 3`` does not:

*Scenario selection.* ``main.py`` resets the env unseeded, so it draws a random
scenario every run. Per-episode TWR across the 12 held-out scenarios ranges from
1.000 down to 0.000 — three of them are flat failures — which makes an unseeded
run a coin flip. Here the reset seeds are pinned to the scenarios below, so the
demo shows the same flights every time.

*Observation width.* The checkpoint was trained before Layer 2 appended
``radiative_forcing_norm``, so it takes 143 inputs and the live env emits 144.
The new field is the last slot, so dropping it recovers the exact observation
the agent was trained on — verified: this reproduces the published benchmark
scenario for scenario, mean TWR 0.394. That is a demo shim for an old
checkpoint, not a compatibility layer; the Layer 2 retrain replaces it.
"""

import shutil
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from sb3_contrib import QRDQN  # noqa: E402

import agents.qrdqn as qrdqn  # noqa: E402
from agents.evaluation import make_scenario_set  # noqa: E402
from environments.envs.balloon_3d_env import (  # noqa: E402
    IDX_RESOURCE_A,
    IDX_RESOURCE_B,
)
from environments.render.pygame_render import PygameRenderer  # noqa: E402
from environments.wrappers.decision_interval import (
    DecisionIntervalWrapper,
)  # noqa: E402

BALLOON = "zero_pressure"
LEGACY_OBS_WIDTH = 143

# Set True if your terminal renders the arrows and blocks below as boxes.
ASCII_ONLY = False

# Keyed by action *effect* (not index): +1 drop ballast -> ascend, 0 hold,
# -1 vent helium -> descend.
GLYPHS = (
    {1: ("^", "BALLAST"), 0: (".", "HOLD"), -1: ("v", "VENT")}
    if ASCII_ONLY
    else {1: ("▲", "BALLAST"), 0: ("·", "HOLD"), -1: ("▼", "VENT")}
)
BAR_FULL, BAR_EMPTY = ("#", "-") if ASCII_ONLY else ("█", "░")
SEP = " | " if ASCII_ONLY else " │ "

# Decisions of history kept in the rolling action tape, and the fewest glyphs
# worth drawing — below this the tape is dropped rather than shown as a stub.
TAPE_LEN = 40
TAPE_MIN = 8
TAPE_LABEL = "recent "

# Held-out scenario indices worth showing, best first, with the TWR each scores
# in `python main.py --benchmark --dim 3`. The other seven are omitted: 1, 2 and
# 5 score 0.000 and 8, 9, 10, 11 are partial holds.
DEMO_SCENARIOS = [(6, 1.000), (0, 0.917), (7, 0.843), (4, 0.778), (3, 0.658)]

# One decision = 60 physics sub-steps = 1 minute of flight. Decisions per second
# of wall clock; the 12-hour mission plays in about two minutes at 6.
SPEED = 6.0

# The env renders every physics sub-step, so a decision costs 60 full-window
# `pygame.display.update()` calls. That is fine on a local display but it is
# ~535 updates/s of an 800x600 window, which saturates an `ssh -X` link and
# drags the demo to a crawl. One frame per decision is all the eye needs at
# demo speed and cuts X traffic 60x.
DRAW_EVERY = 60


def _draw_once_per_decision() -> None:
    """Render only the final sub-step of each decision interval.

    Wraps the renderer rather than touching the env, so this stays a demo
    concern. Note `draw()` also pumps pygame events and applies the fps
    throttle, so decimating it means both now happen once per decision —
    which is why `render_fps` below is set to SPEED rather than 60*SPEED.
    """
    original = PygameRenderer.draw

    def draw(self, state):
        self._demo_frame = getattr(self, "_demo_frame", 0) + 1
        if self._demo_frame % DRAW_EVERY == 0:
            original(self, state)

    PygameRenderer.draw = draw


class StatusLine:
    """A single `\\r`-redrawn line showing the latest decision.

    Redrawn once per decision, so at SPEED=6 it ticks six times a second —
    the point being to make the decision cadence visible rather than to
    report numbers precisely.

    Segments are emitted in priority order and the tail is dropped to fit the
    terminal: a line wider than the terminal wraps, and a wrapped line cannot
    be overwritten by `\\r`, which would turn the status into scrolling spam.

    Falls back to no-op when stdout is not a terminal (piped to a file or a
    log), where `\\r` is unreadable; the hourly prints still carry the run.
    """

    def __init__(self) -> None:
        self.enabled = sys.stdout.isatty()
        self.tape: deque[str] = deque(maxlen=TAPE_LEN)
        self._width = 0  # chars written last time, for clearing

    def _bar(self, frac: float, n: int = 10) -> str:
        filled = int(round(frac * n))
        return f"[{BAR_FULL * filled}{BAR_EMPTY * (n - filled)}] {frac:3.0%}"

    def update(
        self,
        *,
        step: int,
        horizon: int,
        effect: int,
        alt_km: float,
        dist_km: float,
        on_station: bool,
        twr: float,
        ballast: float,
        helium: float,
    ) -> None:
        glyph, label = GLYPHS[effect]
        self.tape.append(glyph)
        if not self.enabled:
            return

        # Highest priority first; the tail is dropped on narrow terminals.
        # `on_station` is padded to equal width in both states so the fields
        # to its right do not jitter as it flips.
        core = [
            f"t+{step // 60:02d}:{step % 60:02d}",
            f"{glyph} {label:<7}",
            f"dist {dist_km:6.2f} km  "
            + ("ON STATION " if on_station else "off station"),
            f"alt {alt_km:5.1f} km",
            f"TWR {twr:.3f}",
        ]

        limit = shutil.get_terminal_size().columns - 1
        line = ""
        for seg in core:
            candidate = seg if not line else f"{line}{SEP}{seg}"
            if len(candidate) > limit:
                break
            line = candidate

        # Resources and progress yield to the tape: they are only added if the
        # tape still gets TAPE_MIN glyphs afterwards. The tape is the one field
        # that shows the *shape* of the control policy, so on a narrow terminal
        # it is worth more than either.
        reserved = len(SEP) + len(TAPE_LABEL) + TAPE_MIN
        for seg in (
            f"bal {ballast * 100:3.0f}% he {helium * 100:3.0f}%",
            self._bar(step / horizon),
        ):
            candidate = f"{line}{SEP}{seg}"
            if len(candidate) + reserved > limit:
                break
            line = candidate

        # The tape takes whatever room is left rather than a fixed width: it
        # grows from empty to TAPE_LEN, so a fixed-width tape fits early on and
        # then silently drops off the end once it outgrows the terminal.
        # Right-aligning the newest glyphs in a constant-width field keeps it
        # on screen and keeps the overall line length stable.
        room = limit - len(line) - len(SEP) - len(TAPE_LABEL)
        if room >= TAPE_MIN:
            width = min(TAPE_LEN, room)
            tape = "".join(self.tape)[-width:].rjust(width)
            line += f"{SEP}{TAPE_LABEL}{tape}"

        # Pad over whatever the previous, possibly longer, line left behind.
        sys.stdout.write("\r" + line.ljust(self._width))
        sys.stdout.flush()
        self._width = len(line)

    def clear(self) -> None:
        """Blank the line so a permanent print can take it over."""
        if self.enabled and self._width:
            sys.stdout.write("\r" + " " * self._width + "\r")
            sys.stdout.flush()
            self._width = 0


def main() -> int:
    _draw_once_per_decision()
    env_config = {**qrdqn._ENV_CONFIG[BALLOON], "balloon_type": BALLOON}
    scenarios = make_scenario_set(
        12, seed=qrdqn.SCENARIO_SEED, held_out=True, config=env_config
    )

    model_path = qrdqn._resolve_model_path(qrdqn.model_save_path(BALLOON))
    if model_path is None:
        print(
            "No checkpoint under src/models/qr_dqn_model/zero_pressure/",
            file=sys.stderr,
        )
        return 1
    model = QRDQN.load(model_path, device=torch.device("cpu"))

    env = DecisionIntervalWrapper(
        gym.make(
            qrdqn._ENV_NAMES[BALLOON],
            render_mode="human",
            dim=3,
            disable_env_checker=True,
            config={**env_config, "render_fps": max(1, int(round(SPEED)))},
        )
    )
    radius = float(env.unwrapped._station_radius)
    horizon = qrdqn.DECISIONS_PER_EPISODE

    print("\nProject Stratus — Layer 1 QR-DQN, 3D zero-pressure")
    print(
        f"{len(DEMO_SCENARIOS)} held-out flights, {horizon} decisions each "
        f"(12 h), station radius {radius / 1000:.0f} km"
    )
    print('Press N (or click "Next") to skip a flight; close the window to stop.\n')

    status = StatusLine()
    scored = []
    for n, (idx, expected) in enumerate(DEMO_SCENARIOS, start=1):
        seed = int(scenarios[idx]["seed"])
        print(
            f"[{n}/{len(DEMO_SCENARIOS)}] heldout-{idx:04d} (seed {seed}) — "
            f"benchmark TWR {expected:.3f}"
        )

        obs, _ = env.reset(seed=seed)
        inside = step = 0
        done = skipped = False
        status.tape.clear()
        while not done:
            action, _ = model.predict(obs[:LEGACY_OBS_WIDTH], deterministic=True)
            effect = int(env.unwrapped._action_lut[int(action)])
            obs, _reward, terminated, truncated, info = env.step(int(action))
            step += 1
            dist = float(info["distance"])
            inside += dist <= radius
            alt = float(env.unwrapped._balloon.pos[2])

            if step % 60 == 0:  # hourly, kept as permanent scrollback
                status.clear()
                print(
                    f"      t+{step // 60:>2}h  alt {alt / 1000:>5.2f} km"
                    f"  dist {dist / 1000:>6.2f} km  TWR {inside / step:.3f}"
                )
            status.update(
                step=step,
                horizon=horizon,
                effect=effect,
                alt_km=alt / 1000,
                dist_km=dist / 1000,
                on_station=dist <= radius,
                twr=inside / step,
                # resource_a/b are the ZP ballast and helium fractions; both
                # sit below index 143 so they survive the observation slice.
                ballast=float(obs[IDX_RESOURCE_A]),
                helium=float(obs[IDX_RESOURCE_B]),
            )
            done = terminated or truncated

            renderer = env.unwrapped.renderer
            if renderer is not None:
                if renderer.quit_requested:
                    status.clear()
                    env.close()
                    return 0
                if renderer.skip_requested:  # "Next (N)" button or the N key
                    renderer.skip_requested = False
                    skipped = True
                    done = True

        status.clear()
        if skipped:
            print(
                f"      -> skipped at t+{step // 60}h (TWR so far {inside / step:.3f})\n"
            )
            continue
        scored.append((idx, inside / horizon))
        print(f"      -> time within radius {inside / horizon:.3f}\n")

    env.close()
    print("Held-out results")
    for idx, twr in scored:
        print(f"  heldout-{idx:04d}   TWR {twr:.3f}")
    print("\nAcross all 12 held-out scenarios (python main.py --benchmark --dim 3):")
    print(
        "  qr_dqn 0.394  |  greedy_wind 0.126  |  random 0.037  |  passive drift 0.037\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
