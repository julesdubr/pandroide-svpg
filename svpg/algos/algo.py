from salina import Agent, get_arguments, get_class, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent

import gym
from gym.wrappers import TimeLimit
from gym.spaces import Box, Discrete

import torch


class Algo:
    def _index(tensor_3d, tensor_2d):
        x, y, z = tensor_3d.size()
        t = tensor_3d.reshape(x * y, z)
        tt = tensor_2d.reshape(x * y)
        v = t[torch.arange(x * y), tt]
        v = v.reshape(x, y)

        return v

    def combine_agents(self):
        raise NotImplementedError()

    def create_particles(self):
        raise NotImplementedError()

    def execute_agent(self):
        raise NotImplementedError()

    def compute_losses(cfg):
        raise NotImplementedError()

    def setup_optimizers(self):
        raise NotImplementedError()

    def make_env(env_name, max_episode_steps):
        """Create the environment using gym:
        - Using hydra to take arguments from a configuration file"""
        return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)
