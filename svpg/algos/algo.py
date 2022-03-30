from salina import Agent, get_arguments, get_class, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent

import gym
from gym.wrappers import TimeLimit
from gym.spaces import Box, Discrete


class Algo:
    def __init__():
        pass

    def combine_agents(self):
        raise NotImplementedError()

    def create_particles(self):
        raise NotImplementedError()

    def execute_agent(self):
        raise NotImplementedError()

    def compute_losses(cfg, workspace, n_particles, epoch, logger, alpha=None):
        raise NotImplementedError()

    def make_env(env_name, max_episode_steps):
        """Create the environment using gym:
        - Using hydra to take arguments from a configuration file"""
        return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


class EnvAgent(AutoResetGymAgent):
    """
    Create the environment agent
    This agent implements N gym environments with auto-reset
    """

    def __init__(self, cfg, pid):
        super().__init__(
            get_class(cfg.algorithm.env),
            get_arguments(cfg.algorithm.env),
            n_envs=cfg.algorithm.n_envs,
            input=f"action{pid}",
            output=f"env{pid}/",
        )
        self.env = instantiate_class(cfg.algorithm.env)

    # This is necessary to create the corresponding RL agent
    def get_obs_and_actions_sizes(self):
        if isinstance(self.env.action_space, Box):
            # Return the size of the observation and action spaces of the environment
            # In the case of a continuous action environment
            return self.env.observation_space.shape[0], self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, Discrete):
            # Return the size of the observation and action spaces of the environment
            return self.env.observation_space.shape[0], self.env.action_space.n
        else:
            print("unknown type of action space", self.env.action_space)
            return None
