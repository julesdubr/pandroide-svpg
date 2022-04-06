from salina import Agent, get_arguments, get_class, instantiate_class

import torch
import torch.nn as nn
from torch.distributions.normal import Normal


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

class ContinousCriticAgent(Agent):
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
            input_size, output_size, activation="SiLU", **get_arguments(kwargs["model"])
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("critic", t), critic)
