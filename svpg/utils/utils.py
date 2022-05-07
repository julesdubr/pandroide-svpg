from re import A
import numpy as np
import torch as th

import os
from pathlib import Path


def save_algo_data(algo, directory):
    directory = str(directory) + algo.__class__.__name__

    if not os.path.exists(directory):
        os.makedirs(directory)

    rewards = np.array(
        [[r.cpu() for r in agent_reward] for agent_reward in algo.rewards.values()]
    )

    rewards_path = Path(directory + "/rewards.npy")
    with open(rewards_path, "wb") as f:
        np.save(f, rewards)

    action_path = Path(directory + "/action_agents")
    critic_path = Path(directory + "/critic_agents")

    if not os.path.exists(action_path):
        os.makedirs(action_path)
    if not os.path.exists(critic_path):
        os.makedirs(critic_path)

    for i, (a_agent, c_agent) in enumerate(zip(algo.action_agents, algo.critic_agents)):
        th.save(a_agent, str(action_path) + f"/action_agent{i}.pt")
        th.save(c_agent, str(critic_path) + f"/critic_agent{i}.pt")


def load_algo_data(directory):
    directory = str(directory)

    with open(directory + "/rewards.npy", "rb") as f:
        rewards = np.load(f, allow_pickle=True)

    action_agents, action_path = [], directory + "/action_agents"
    critic_agents, critic_path = [], directory + "/critic_agents"

    for i in range(16):
        action_agent = th.load(action_path + f"/action_agent{i}.pt").cpu()
        action_agents.append(action_agent)
        critic_agents.append(th.load(critic_path + f"/critic_agent{i}.pt").cpu())

    return action_agents, critic_agents, rewards
