import numpy as np

from salina.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from salina import instantiate_class, get_arguments, get_class

import gym
import my_gym
import gym_cartpole_swingup
from svpg.utils import rllab_gym
from rllab.spaces import Discrete, Box


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env, lower_bound=None, upper_bound=None):
        super().__init__(env)
        if lower_bound is None and upper_bound is None:
            self.lower_bound, self.upper_bound = env.action_space.bounds
        else:
            self.lower_bound, self.upper_bound = lower_bound, upper_bound

    def action(self, action):
        # scaled_action = self.lower_bound + (action + 1) * 0.5 * (
        #     self.upper_bound - self.lower_bound
        # )
        # scaled_action = np.clip(scaled_action, self.lower_bound, self.upper_bound)
        return np.clip(action, self.lower_bound, self.upper_bound)


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, alpha=0.001, epsilon=1e-8):
        super().__init__(env)
        self.alpha = alpha
        self.epsilon = epsilon
        self.means = np.zeros(env.observation_space.flat_dim)
        self.vars = np.ones(env.observation_space.flat_dim)
        self.wrapped_env = env

    def update_estimate(self, obs):
        flat_obs = self.wrapped_env.observation_space.flatten(obs)
        one_alpha = 1 - self.alpha
        self.means = one_alpha * self.means + self.alpha * flat_obs
        self.vars = one_alpha * self.vars + self.alpha * (flat_obs - self.means) ** 2

    def observation(self, obs):
        self.update_estimate(obs)
        return (obs - self.means) / (np.sqrt(self.vars) + self.epsilon)


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, alpha=1e-3, epsilon=1e-8) -> None:
        super().__init__(env)
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean = 0
        self.var = 1

    def update_estimate(self, reward):
        self.mean = (1 - self.alpha) * self.mean + self.alpha * reward
        self.var = (1 - self.alpha) * self.var + self.alpha * (reward - self.mean) ** 2

    def reward(self, reward: float) -> float:
        self.update_estimate(reward)
        return (reward - self.mean) / (np.sqrt(self.var) + self.epsilon)


def make_gym_env(env_name, wrap_action=True, wrap_reward=True, wrap_obs=True):
    return gym.make(env_name)
    # env = gym.make(env_name)
    # if wrap_action:
    #     env = ActionWrapper(env)
    # if wrap_reward:
    #     env = RewardWrapper(env)
    # if wrap_obs:
    #     env = ObservationWrapper(env)
    # return env


def get_env_infos(env):
    action_dim, state_dim = 0, 0
    continuous_action, continuous_state = False, False

    if env.is_continuous_action() or isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
        continuous_action = True
    elif env.is_discrete_action() or isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
    if env.is_continuous_state() or isinstance(env.observation_space, Box):
        state_dim = env.observation_space.shape[0]
        continuous_state = True
    elif env.is_discrete_state() or isinstance(env.observation_space, Discrete):
        state_dim = env.observation_space.n

    return (continuous_state, state_dim), (continuous_action, action_dim)


class AutoResetEnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, n_envs, **kwargs):
        args = get_arguments(cfg.gym_env) | kwargs
        super().__init__(get_class(cfg.gym_env), args, n_envs)
        env = instantiate_class(cfg.gym_env)
        env.seed(cfg.algorithm.seed)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env


class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments without auto-reset
    def __init__(self, cfg, n_envs, **kwargs):
        args = get_arguments(cfg.gym_env) | kwargs
        super().__init__(get_class(cfg.gym_env), args, n_envs)
        env = instantiate_class(cfg.gym_env)
        env.seed(cfg.algorithm.seed)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        del env
