import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal

from salina.agent import Agent


class CActionAgent(Agent):
    def __init__(self, output_size, model):
        super().__init__(name="action_agent")
        # Model input and output size
        # Model for estimating the mean
        self.model = model
        # The deviation is estimated by a vector
        self.std_param = nn.parameter.Parameter(th.randn(output_size, 1))
        self.soft_plus = nn.Softplus()

    def forward(self, t, stochastic, replay=False, **kwargs):
        if "observation" in kwargs:
            observation = kwargs["observation"]
        else:
            observation = self.get(("env/env_obs", t))
        mean = self.model(observation)
        dist = Normal(mean, self.soft_plus(self.std_param))
        entropy = dist.entropy().squeeze(-1)

        if stochastic:
            action = th.tanh(dist.sample())
        else:
            action = th.tanh(mean)

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
