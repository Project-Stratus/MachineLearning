"""
quick test
Run each dimensional mode for a few seconds with random actions.
"""

import time
import pygame
from environments.envs.balloon_3d_env import Balloon3DEnv


def run(dim, steps=300, human_render=True):
    print(f"\n=== Testing dim={dim} ===")
    env = Balloon3DEnv(dim=dim, render_mode="human" if human_render else None)

    obs, info = env.reset(seed=42)
    print("Reset → obs shape:", obs.shape)

    for t in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            print(f"Episode finished at step {t}  |  reward {reward:.1f}")
            break

        if human_render:
            time.sleep(0.01)

        if pygame.event.peek(pygame.QUIT):
            env.close()
            return

    env.close()


if __name__ == "__main__":
    for dim in (1, 2, 3):
        run(dim, steps=100)
