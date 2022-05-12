from salina.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent

import numpy as np

import gym
import my_gym

from svpg import rllab_env_wrapper


class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env, lower_bound=None, upper_bound=None):
        super().__init__(env)
        if lower_bound is None and upper_bound is None:
            self.lower_bound, self.upper_bound = env.action_space.bounds
        else:
            self.lower_bound, self.upper_bound = lower_bound, upper_bound

    def action(self, action):
        scaled_action = self.lower_bound + (action + 1) * 0.5 * (self.upper_bound - self.lower_bound)
        scaled_action = np.clip(scaled_action, self.lower_bound, self.upper_bound)

        return scaled_action

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
        self.means = (1 - self.alpha) * self.means + self.alpha * flat_obs
        self.vars = (1 - self.alpha) * self.vars + self.alpha * (flat_obs - self.means) ** 2
    
    def observation(self, obs):
        self.update_estimate(obs)

        return (obs - self.means) / (np.sqrt(self.vars) + self.epsilon)


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, alpha=0.001, epsilon=1e-8) -> None:
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


def make_env(env_name):
    """
    Create the environment using gym:
    - Using hydra to take arguments from a configuration file
    """
    env = gym.make(env_name)

    return ObservationWrapper(RewardWrapper(ActionWrapper(env)))

class AutoResetEnvAgent(AutoResetGymAgent):
    """
    Create the environment agent.
    This agent implements N gym environments with auto-reset.
    """

    def __init__(self, env_name, n_envs, make_env_fn=make_env):
        super().__init__(
            make_env_fn=make_env_fn,
            make_env_args={
                "env_name": env_name
            },
            n_envs=n_envs,
        )


class NoAutoResetEnvAgent(NoAutoResetGymAgent):
    def __init__(self, env_name, n_envs, make_env_fn=make_env):
        super().__init__(
            make_env_fn=make_env_fn,
            make_env_args={
                "env_name": env_name
            },
            n_envs=n_envs,
        )
