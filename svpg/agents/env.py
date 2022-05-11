from salina.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from salina import instantiate_class, get_arguments, get_class

import gym
import my_gym
import gym_cartpole_swingup

from rllab.spaces import Discrete, Box

from svpg.utils import rllab_env_wrapper


def make_gym_env(env_name):
    """
    Create the environment using gym:
    - Using hydra to take arguments from a configuration file
    """
    return gym.make(env_name)


def get_env_infos(env):
    action_dim = 0
    state_dim = 0
    continuous_action = False
    if env.is_continuous_action() or isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
        continuous_action = True
    elif env.is_discrete_action() or isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
    if env.is_continuous_state() or isinstance(env.observation_space, Box):
        state_dim = env.observation_space.shape[0]
    elif env.is_discrete_state() or isinstance(env.observation_space, Discrete):
        state_dim = env.observation_space.n

    return state_dim, action_dim, continuous_action


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
