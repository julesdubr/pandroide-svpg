import torch as th

from salina import Agent


class ActionAgent(Agent):
    def __init__(self, model):
        super().__init__()
        # Model
        self.model = model

    def forward(self, t, stochastic, **kwargs):
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

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("critic", t), critic)
