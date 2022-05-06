import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from salina.agent import Agent


class CActionAgent(Agent):
    def __init__(self, output_size, model):
        super().__init__(name="action_agent")
        # Model input and output size
        # Model for estimating torche mean
        self.model = model
        # The deviation is estimated by a vector
        init_variance = torch.randn(output_size, 1)
        self.std_param = nn.parameter.Parameter(init_variance)
        self.soft_plus = nn.Softplus()

    def forward(self, t, stochastic, replay=False, **kwargs):
        observation = self.get(("env/env_obs", t))
        mean = self.model(observation)
        dist = Normal(mean, self.soft_plus(self.std_param))
        entropy = dist.entropy().squeeze(-1)

        if stochastic:
            action = dist.sample()
        else:
            action = mean

        if t == -1:
            return action

        action_logprobs = dist.log_prob(action).sum(axis=-1)
        
        if not replay:
            self.set(("action", t), action)

        self.set(("action_logprobs", t), action_logprobs)
        self.set(("entropy", t), entropy)


class CCriticAgent(Agent):
    """
    CriticAgent:
    - A one hidden layer neural network which takes an observation as input and whose
      output is the value of this observation.
    - It thus implements a V(s)  function
    """

    def __init__(self, model):
        super().__init__()
        # Model
        self.model = model

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("critic", t), critic)
