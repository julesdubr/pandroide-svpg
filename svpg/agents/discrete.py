import torch as th

from salina import TAgent


class ActionAgent(TAgent):
    def __init__(self, model):
        super().__init__()
        # Model
        self.model = model

    def forward(self, t, stochastic, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = th.softmax(scores, dim=-1)

        if stochastic:
            action = th.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        entropy = th.distributions.Categorical(probs).entropy()
        probs = probs[th.arange(probs.size()[0]), action]

        self.set(("action", t), action)
        self.set(("action_logprobs", t), probs.log())
        self.set(("entropy", t), entropy)


class CriticAgent(TAgent):
    """
    CriticAgent:
    - A one hidden layer neural network which takes an observation as input and whose
      output is the value of this observation.
    - It thus implements a V(s)  function
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.model(observation).squeeze(-1)
        self.set(("critic", t), critic)
