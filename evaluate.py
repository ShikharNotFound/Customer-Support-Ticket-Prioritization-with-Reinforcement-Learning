import torch
import numpy as np
from server.meta_env_environment import TicketEnv
from server.dqn_agent import DQNAgent

env = TicketEnv()
agent = DQNAgent(state_size=50, action_size=10)
agent.q_network.load_state_dict(torch.load("dqn_model.pth"))
agent.epsilon = 0.0  # no exploration

obs = env.reset()
state = env.get_state_vector()
total_reward = 0
done = False
step = 0
while not done and step < 200:
    action = agent.act(state, eval_mode=True)
    next_obs, reward, done, info = env.step(action)
    next_state = env.get_state_vector()
    state = next_state
    total_reward += reward
    step += 1

print(f"Trained agent total reward: {total_reward:.2f}")
print(f"SLA violations: {info.get('violations', 0)}")