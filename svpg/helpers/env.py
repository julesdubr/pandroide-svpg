from salina import get_arguments, get_class, instantiate_class
from salina.agents.gyma import AutoResetGymAgent

from gym.spaces import Box, Discrete


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
