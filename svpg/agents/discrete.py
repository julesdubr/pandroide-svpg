import torch
from salina import Agent, get_arguments, get_class, instantiate_class


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
