from salina import Agent, get_arguments, get_class, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent

from gym.spaces import Box, Discrete

import torch, torch.nn as nn


class ProbAgent(Agent):
    def __init__(self, observation_size, hidden_size, n_actions, pid):
        # We need to add the pid of the particle to its prob_agent name so
        # that we can synchronize the acquisition_agent of each particle to
        # the prob_agent corresponding
        super().__init__(name=f"prob_agent{pid}")
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.pid = pid

    def forward(self, t, **kwargs):
        observation = self.get((f"env{self.pid}/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        self.set((f"action_probs{self.pid}", t), probs)


class ActionAgent(Agent):
    def __init__(self, pid):
        super().__init__()
        self.pid = pid

    def forward(self, t, stochastic, **kwargs):
        probs = self.get((f"action_probs{self.pid}", t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set((f"action{self.pid}", t), action)

class CriticAgent(Agent):
    def __init__(self, observation_size, hidden_size, pid):
        super().__init__()
        self.critic_model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.pid = pid

    def forward(self, t, **kwargs):
        observation = self.get((f"env{self.pid}/env_obs", t))
        critic = self.critic_model(observation).squeeze(-1)
        self.set((f"critic{self.pid}", t), critic)

class EnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, pid):
        super().__init__(
            get_class(cfg.algorithm.env),
            get_arguments(cfg.algorithm.env),
            n_envs=cfg.algorithm.n_envs,
            input=f"action{pid}",
            output=f"env{pid}/",
        )
        self.env = instantiate_class(cfg.algorithm.env)

    # Return the size of the observation and action spaces of the env
    def get_obs_and_actions_sizes(self):
        action_space = None

        if isinstance(self.env.action_space, Box):
            action_space = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, Discrete):
            action_space = self.env.action_space.n

        return self.env.observation_space.shape[0], action_space


def create_acquisition_agent(cfg, env_agent, pid):
    # Get info on the environment
    observation_size, n_actions = env_agent.get_obs_and_actions_sizes()
    del env_agent.env

    acq_env_agent = EnvAgent(cfg, pid)

    prob_agent = ProbAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions, pid
    )

    action_agent = ActionAgent(pid)

    # Combine env and acquisition agents
    # We'll combine the acq_agents of all particle into a single TemporalAgent later
    acq_agent = Agents(acq_env_agent, prob_agent, action_agent)

    critic_agent = CriticAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, pid
    )

    return acq_agent, prob_agent, critic_agent

def combine_agents(cfg, particles):
    # Combine all acquisition agent of all particle in a unique TemporalAgent.
    # This will help us to avoid using a loop explicitly to execute all these agents
    # (these agents will still be executed by a for loop by SaliNa)
    acq_agents = TemporalAgent(
        Agents(*[particle["acq_agent"] for particle in particles])
    )

    # Set the seed
    acq_agents.seed(cfg.algorithm.env_seed)

    # We also combine all the critic_agent of all particle into a unique TemporalAgent
    tcritic_agent = TemporalAgent(
        Agents(*[particle["critic_agent"] for particle in particles])
    )

    return acq_agents, tcritic_agent