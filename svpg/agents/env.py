from salina import get_arguments, get_class
from salina.agents.gyma import AutoResetGymAgent, GymAgent

import gym
from gym.wrappers import TimeLimit


def make_env(env_name, max_episode_steps):
    """
    Create the environment using gym:
    - Using hydra to take arguments from a configuration file
    """
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)

class EnvAgentAutoReset(AutoResetGymAgent):
    """
    Create the environment agent.
    This agent implements N gym environments with auto-reset.
    """

    def __init__(self, env_name, max_episode_steps, n_envs, make_env_fn=make_env):
        super().__init__(
            make_env_fn=make_env_fn,
            make_env_args={"env_name": env_name, "max_episode_steps": max_episode_steps},
            n_envs=n_envs
        )
    


class EnvAgent(GymAgent):
    def __init__(self, env_name, max_episode_steps, n_envs, make_env_fn=make_env):
        super().__init__(
            make_env_fn=make_env_fn,
            make_env_args={"env_name": env_name, "max_episode_steps": max_episode_steps},
            n_envs=n_envs
        )
