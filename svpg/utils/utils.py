import numpy as np
import torch as th

import os


def save_algo_data(action_agents, critic_agents, rewards, directory):
    directory = str(directory)

    rewards = np.array(
        [[r.cpu() for r in agent_reward] for agent_reward in rewards.values()]
    )

    with open(directory + "/rewards.npy", "wb") as f:
        np.save(f, rewards)

    action_path = directory + "/action_agents"
    critic_path = directory + "/critic_agents"

    if not os.path.exists(action_path):
        os.makedirs(action_path)
    if not os.path.exists(critic_path):
        os.makedirs(critic_path)

    for i, (a_agent, c_agent) in enumerate(zip(action_agents, critic_agents)):
        th.save(a_agent, action_path + f"/action_agent{i}.pt")
        th.save(c_agent, critic_path + f"/critic_agent{i}.pt")


def load_algo_data(directory):
    directory = str(directory)

    with open(directory + "/rewards.npy", "rb") as f:
        rewards = np.load(f, allow_pickle=True)

    action_agents, action_path = [], directory + "/action_agents"
    critic_agents, critic_path = [], directory + "/critic_agents"

    for i in range(rewards.shape[0]):
        action_agents.append(th.load(action_path + f"/action_agent{i}.pt").cpu())
        critic_agents.append(th.load(critic_path + f"/critic_agent{i}.pt").cpu())

    return action_agents, critic_agents, rewards
