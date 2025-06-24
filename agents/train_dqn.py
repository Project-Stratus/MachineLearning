"""
train_dqn.py

Trains a DQN agent on the BalloonEnv.
The agent learns to apply a vertical control force to keep the balloon near its target altitude.
Additional diagnostic plots are saved to show the wind field, training rewards, epsilon decay, and the test episode trajectory.
At the end, an animation is generated to show how the trained policy drives the balloon.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from environments.envs.balloon_env import Balloon2DEnv, X_RANGE, Y_RANGE, x_forces, y_forces, fx_grid, fy_grid, x_edges, y_edges    # noqa

from agents.dqn_agent import DQNAgent


def train(num_episodes=500, target_update=10):
    env = Balloon2DEnv()
    state_dim = env.observation_space.shape[0]  # [y, vy]
    action_dim = env.action_space.n             # 3 actions: down, none, up
    agent = DQNAgent(state_dim, action_dim)

    rewards_history = []
    epsilon_history = []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            episode_reward += reward

        rewards_history.append(episode_reward)
        epsilon_history.append(agent.epsilon)
        if episode % target_update == 0:
            agent.update_target()

        print(f"Episode {episode+1}/{num_episodes}, Total Reward: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    env.close()

    # Save reward history plot.
    plt.figure()
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward History")
    plt.savefig("reward_history.png")
    plt.close()

    # Save epsilon decay plot.
    plt.figure()
    plt.plot(epsilon_history)
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Epsilon Decay Over Episodes")
    plt.savefig("epsilon_decay.png")
    plt.close()

    # Save the trained model.
    torch.save(agent.policy_net.state_dict(), "dqn_balloon_model.pth")

    # Save the wind field image using the environment's built-in function.
    env_vis = Balloon2DEnv()
    env_vis.save_wind_field("wind_field.png")
    env_vis.close()

    # Run one test episode to record the balloon's trajectory.
    test_states = []
    state, info = env.reset()
    done = False
    while not done:
        test_states.append((env.balloon.x, env.balloon.y))
        action = agent.select_action(state)
        state, _, done, _ = env.step(action)
    env.close()

    # Plot the trajectory over the wind field.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("Balloon Trajectory Over Wind Field")
    ax.set_xlabel("X position (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_xlim(X_RANGE)
    ax.set_ylim(Y_RANGE)
    # Draw the wind field.
    # q = ax.quiver(x_forces, y_forces, fx_grid, fy_grid, color='blue', alpha=0.5)
    test_states_arr = np.array(test_states)
    ax.plot(test_states_arr[:, 0], test_states_arr[:, 1], marker='o', color='red', linewidth=2, markersize=4)
    plt.savefig("balloon_trajectory.png")
    plt.close()

    # ---- Create an animation showing how the trained policy drives the balloon ----
    fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
    ax_anim.set_title("Balloon Animation Under Trained Policy")
    ax_anim.set_xlabel("X position (m)")
    ax_anim.set_ylabel("Altitude (m)")
    ax_anim.set_xlim(X_RANGE)
    ax_anim.set_ylim(Y_RANGE)
    # Draw the wind field as background.
    ax_anim.quiver(x_forces, y_forces, fx_grid, fy_grid, color='blue', alpha=0.5)
    # Create a circle to represent the balloon.
    balloon_circle = plt.Circle(test_states[0], 50, color='red')
    ax_anim.add_patch(balloon_circle)

    def animate_balloon(i):
        x, y = test_states[i]
        balloon_circle.center = (x, y)
        return (balloon_circle,)

    anim = animation.FuncAnimation(fig_anim, animate_balloon,
                                   frames=len(test_states), interval=50, blit=True)
    # Save the animation as an MP4 file. Change writer or filename as desired.
    anim.save("balloon_animation.gif", writer="pillow")
    print("Animation saved as 'balloon_animation.gif'. Update to .mp4 later.")
    plt.close(fig_anim)

    return agent


if __name__ == "__main__":
    trained_agent = train()
