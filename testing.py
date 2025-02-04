import gymnasium as gym
import loon_v0
from loon_v0.wrappers.policy import MyPolicy

if __name__ == "__main__":

    env = gym.make("loon_v0/LoonEnv-v0", render_mode="human")
    observation, info = env.reset()

    policy = MyPolicy(env.action_space)

    episode_over = False
    current_action = 0

    while not episode_over:
        action = policy.get_action(observation)

        next_observation, reward, terminated, truncated, info = env.step(action)

        observation = next_observation

        episode_over = terminated or truncated

        new_action = action[0]
        if new_action != current_action:
            print(f"Action: {new_action}")
            current_action = new_action

    env.close()
