import torch as th

from salina.agent import Agent


class ActionAgent(Agent):
    def __init__(self, model):
        super().__init__(name="action_agent")
        # Model
        self.model = model

    def forward(self, t, stochastic, replay=False, **kwargs):
        if "observation" in kwargs:
            observation = kwargs["observation"]
        else:
            observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = th.softmax(scores, dim=-1)

        if stochastic:
            action = th.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        if t == -1:
            return action

        entropy = th.distributions.Categorical(probs).entropy()
        logprobs = probs[th.arange(probs.size()[0]), action].log()

        if not replay:
            self.set(("action", t), action)
        self.set(("action_logprobs", t), logprobs)
        self.set(("entropy", t), entropy)


class CriticAgent(Agent):
    """
    CriticAgent:
    - A one hidden layer neural network which takes an observation as input and whose
      output is the value of this observation.
    - It thus implements a V(s) function
    """


import torch as th
import torch.nn as nn

from salina.agent import Agent

from .model import build_mlp


class ActionAgent(Agent):
    def __init__(self, state_dim, hidden_layers, n_action):
        super().__init__(name="action_agent")
        # Model
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [n_action], activation=nn.ReLU()
        )

    def forward(self, t, stochastic, replay=False, **kwargs):
        if "observation" in kwargs:
            observation = kwargs["observation"]
        else:
            observation = self.get(("env/env_obs", t))

        scores = self.model(observation)
        probs = th.softmax(scores, dim=-1)

        if stochastic:
            action = th.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        if t == -1:
            return action

        logprobs = probs[th.arange(probs.size()[0]), action].log()
        self.set(("action_logprobs", t), logprobs)
        entropy = th.distributions.Categorical(probs).entropy()
        self.set(("entropy", t), entropy)

        if not replay:
            self.set(("action", t), action)


class CriticAgent(Agent):
    """
    CriticAgent:
    - A one hidden layer neural network which takes an observation as input and whose
      output is the value of this observation.
    - It thus implements a V(s) function
    """

    def __init__(self, state_dim, hidden_layers):
        super().__init__()
        self.model = build_mlp(
            [state_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("critic", t), critic)
