import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal

from salina.agent import Agent

from .model import build_mlp


class ContinuousActionAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim, eval=False, **kwargs):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(layers, activation=nn.ReLU())
        # The deviation is estimated by a vector
        init_variance = th.randn(action_dim, 1)
        self.std_param = nn.parameter.Parameter(init_variance)
        self.soft_plus = nn.Softplus()

    def forward(self, t, stochastic, **kwargs):
        obs = self.get(("env/env_obs", t))
        mean = self.model(obs)

        # std must be positive
        dist = Normal(mean, th.exp(self.soft_plus(self.std_param)))
        self.set(("entropy", t), dist.entropy())

        if stochastic:
            action = dist.sample()  # valid actions are supposed to be in [-1,1] range
        else:
            action = mean  # valid actions are supposed to be in [-1,1] range

        logp_pi = dist.log_prob(action).sum(axis=-1)
        self.set(("action", t), action)
        self.set(("action_logprobs", t), logp_pi)

    def predict_action(self, obs, stochastic):
        mean = self.model(obs)
        dist = Normal(mean, self.soft_plus(self.std_param))
        if stochastic:
            action = dist.sample()  # valid actions are supposed to be in [-1,1] range
        else:
            action = mean  # valid actions are supposed to be in [-1,1] range
        return action
