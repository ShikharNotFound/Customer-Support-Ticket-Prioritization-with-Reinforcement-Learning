# meta_env/train.py
import numpy as np
import torch
from server.meta_env_environment import TicketEnv
from server.dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(episodes=500, max_steps=200):
    env = TicketEnv(max_tickets=10, episode_steps=max_steps)
    agent = DQNAgent(state_size=50, action_size=10)
    rewards_per_episode = []
    sla_violations_per_episode = []

    for ep in tqdm(range(episodes)):
        obs = env.reset()
        state = env.get_state_vector()
        total_reward = 0
        total_violations = 0
        for step in range(max_steps):
            action = agent.act(state)
            next_obs, reward, done, info = env.step(action)
            next_state = env.get_state_vector()
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            total_reward += reward
            total_violations += info.get("violations", 0)
            if done:
                break
        rewards_per_episode.append(total_reward)
        sla_violations_per_episode.append(total_violations)

        if ep % 10 == 0:
            agent.update_target()

        if ep % 50 == 0:
            print(f"Episode {ep}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(rewards_per_episode)
    plt.title("Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.subplot(1,2,2)
    plt.plot(sla_violations_per_episode)
    plt.title("SLA Violations per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Violations")
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()

    torch.save(agent.q_network.state_dict(), "dqn_model.pth")
    print("Training complete. Model saved as dqn_model.pth")

if __name__ == "__main__":
    train()