from salina.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from salina import instantiate_class, get_arguments, get_class

import gym
import my_gym
from gym.wrappers import TimeLimit

from svpg.utils import rllab_env_wrapper


def make_gym_env(max_episode_steps, env_name):
    """
    Create the environment using gym:
    - Using hydra to take arguments from a configuration file
    """
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


class AutoResetEnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs)
        env = instantiate_class(cfg.gym_env)
        env.seed(cfg.algorithm.seed)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env


class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments without auto-reset
    def __init__(self, cfg, n_envs):
        super().__init__(get_class(cfg.gym_env), get_arguments(cfg.gym_env), n_envs)
        env = instantiate_class(cfg.gym_env)
        env.seed(cfg.algorithm.seed)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env
