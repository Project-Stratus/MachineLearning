import gymnasium as gym
import loon_v0


if __name__ == "__main__":

    env = gym.make("loon_v0/LoonEnv-v0", render_mode="human")
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    env.close()