from salina import get_arguments, get_class, instantiate_class
from salina.agents.gyma import AutoResetGymAgent, GymAgent

import gym
from gym.wrappers import TimeLimit


class EnvAgentAutoReset(AutoResetGymAgent):
    """
    Create the environment agent.
    This agent implements N gym environments with auto-reset.
    """

    def __init__(self, cfg):
        super().__init__(
            get_class(cfg.algorithm.env),
            get_arguments(cfg.algorithm.env),
            n_envs=cfg.algorithm.n_envs,
        )
        self.env = instantiate_class(cfg.algorithm.env)


class EnvAgent(GymAgent):
    def __init__(self, cfg):
        super().__init__(
            get_class(cfg.algorithm.env),
            get_arguments(cfg.algorithm.env),
            n_envs=cfg.algorithm.n_envs,
        )
        self.env = instantiate_class(cfg.algorithm.env)


def make_env(env_name, max_episode_steps):
    """
    Create the environment using gym:
    - Using hydra to take arguments from a configuration file
    """
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)
