from salina import Agent, get_arguments, get_class, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent

import torch
import torch.nn as nn

import gym
from gym.wrappers import TimeLimit
from gym.spaces import Box, Discrete


class ActionAgent(Agent):
    def __init__(self, **kwargs):
        super.__init__()
        # Environment
        env = instantiate_class(kwargs["env"])
        # Model input and output size
        input_size = env.observation_space.shape[0]
        if isinstance(env.action_space, Box):
            output_size = env.action_space.shape[0]
        elif isinstance(env.action_space, Discrete):
            output_size = env.action_space.n
        # Model
        model_generator = get_class(kwargs["model"])
        self.model = model_generator(input_size, output_size)

    def forward(self, t, stochastic, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action_probs", t), probs)
        self.set(("action", t), action)


class CriticAgent(Agent):
    """
    CriticAgent:
    - A one hidden layer neural network which takes an observation as input and whose output is the value of this observation.
    - It thus implements a  V(s)  function
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Environment
        env = instantiate_class(kwargs["env"])
        # Model input and output size
        input_size = env.observation_space.shape[0]
        output_size = 1
        # Model
        model_generator = get_class(kwargs["model"])
        self.model = model_generator(input_size, output_size)

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.critic_model(observation).squeeze(-1)
        self.set(("critic", t), critic)


def make_env(env_name, max_episode_steps):
    """
    Create the environment using gym:
    - Using hydra to take arguments from a configuration file
    """
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


class EnvAgent(AutoResetGymAgent):
    """
    Create the environment agent
    This agent implements N gym environments with auto-reset
    """

    def __init__(self, cfg):
        super().__init__(
            get_class(cfg.algorithm.env),
            get_arguments(cfg.algorithm.env),
            n_envs=cfg.algorithm.n_envs
        )