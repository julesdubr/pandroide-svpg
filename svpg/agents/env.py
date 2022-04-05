from salina import get_arguments, get_class, instantiate_class
from salina.agents.gyma import AutoResetGymAgent, GymAgent

import gym
from gym.wrappers import TimeLimit


class EnvAgentAutoReset(AutoResetGymAgent):
    """
    Create the environment agent.
    This agent implements N gym environments with auto-reset.
    """

    def __init__(self, **kwargs):
        super().__init__(
            make_env_fn=get_class(kwargs["env"]),
            make_env_args=get_arguments(kwargs["env"]),
            n_envs=kwargs["n_envs"],
        )
        self.env = instantiate_class(kwargs["env"])


class EnvAgent(GymAgent):
    def __init__(self, **kwargs):
        super().__init__(
            make_env_fn=get_class(kwargs["env"]),
            make_env_args=get_arguments(kwargs["env"]),
            n_envs=kwargs["n_envs"],
        )


def make_env(env_name, max_episode_steps):
    """
    Create the environment using gym:
    - Using hydra to take arguments from a configuration file
    """
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)
