from salina import Agent, get_arguments, get_class, instantiate_class
from salina.agents.gyma import AutoResetGymAgent, GymAgent

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

import gym
from gym.wrappers import TimeLimit


class ActionAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        # Environment
        env = instantiate_class(kwargs["env"])
        # Model input and output size
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.n
        # Model
        self.model = get_class(kwargs["model"])(
            input_size, output_size, **get_arguments(kwargs["model"])
        )

    def forward(self, t, stochastic, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)

        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        entropy = torch.distributions.Categorical(probs).entropy()
        probs = probs[torch.arange(probs.size()[0]), action]

        self.set(("action", t), action)
        self.set(("action_logprobs", t), probs.log())
        self.set(("entropy", t), entropy)


class ContinuousActionAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()
        # Environment
        env = instantiate_class(kwargs["env"])
        # Model input and output size
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        # Model for estimating the mean
        self.model = get_class(kwargs["model"])(
            input_size, output_size, **get_arguments(kwargs["model"])
        )
        # The deviation is estimated by a vector
        self.std_param = nn.parameter.Parameter(torch.randn(output_size, 1))
        self.soft_plus = nn.Softplus()

    def forward(self, t, stochastic, **kwargs):
        observation = self.get(("env/env_obs", t))
        mean = self.model(observation)
        dist = Normal(mean, self.soft_plus(self.std_param))
        self.set(("entropy", t), dist.entropy().squeeze(-1))

        if stochastic:
            action = torch.tanh(dist.sample())
        else:
            action = torch.tanh(mean)

        action_logprobs = dist.log_prob(action).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), action_logprobs)


class CriticAgent(Agent):
    """
    CriticAgent:
    - A one hidden layer neural network which takes an observation as input and whose
      output is the value of this observation.
    - It thus implements a V(s)  function
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Environment
        env = instantiate_class(kwargs["env"])
        # Model input and output size
        input_size = env.observation_space.shape[0]
        output_size = 1
        # Model
        self.model = get_class(kwargs["model"])(
            input_size, output_size, **get_arguments(kwargs["model"])
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("critic", t), critic)


def make_env(env_name, max_episode_steps):
    """
    Create the environment using gym:
    - Using hydra to take arguments from a configuration file
    """
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


class EnvAgentAutoReset(AutoResetGymAgent):
    """
    Create the environment agent
    This agent implements N gym environments with auto-reset
    """

    def __init__(self, **kwargs):
        super().__init__(
            make_env_fn=get_class(kwargs["env"]),
            make_env_args=get_arguments(kwargs["env"]),
            n_envs=kwargs["n_envs"],
        )


class EnvAgent(GymAgent):
    def __init__(self, **kwargs):
        super().__init__(
            make_env_fn=get_class(kwargs["env"]),
            make_env_args=get_arguments(kwargs["env"]),
            n_envs=kwargs["n_envs"],
        )


def make_model(input_size, output_size, **kwargs):
    # print(kwargs)
    hidden_size = list(kwargs.values())
    # print(hidden_size)
    if len(hidden_size) > 1:
        hidden_layers = [
            [nn.Linear(hidden_size[i], hidden_size[i + 1]), nn.ReLU()]
            for i in range(0, len(hidden_size) - 1)
        ]

        hidden_layers = [l for layers in hidden_layers for l in layers]
    else:
        hidden_layers = [nn.Identity()]

    return nn.Sequential(
        nn.Linear(input_size, hidden_size[0]),
        nn.ReLU(),
        *hidden_layers,
        nn.Linear(hidden_size[-1], output_size)
    )
