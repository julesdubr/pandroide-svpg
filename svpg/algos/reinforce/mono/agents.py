import torch
import torch.nn as nn

from salina import TAgent
from salina.agents import Agents, TemporalAgent

from svpg.algos.a2c.mono.agents import EnvAgent


class REINFORCEAgent(TAgent):
    def __init__(self, observation_size, hidden_size, n_actions, pid):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.critic_model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.pid = pid

    def forward(self, t, stochastic, **kwargs):
        observation = self.get((f"env{self.pid}/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        critic = self.critic_model(observation).squeeze(-1)
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set((f"action{self.pid}", t), action)
        self.set((f"action_probs{self.pid}", t), probs)
        self.set((f"baseline{self.pid}", t), critic)


def create_reinforce_agent(cfg, env_agent, pid):
    # Get info on the environment
    observation_size, n_actions = env_agent.get_obs_and_actions_sizes()
    del env_agent.env

    acq_env_agent = EnvAgent(cfg, pid)

    reinforce_agent = REINFORCEAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions, pid
    )

    acq_agent = Agents(acq_env_agent, reinforce_agent)

    return acq_agent, reinforce_agent


def create_particles(cfg, n_particles, env_agents):
    acq_agents = list()
    reinforce_agents = list()

    for i in range(n_particles):
        # Create A2C agent for all particles
        acq_agent, reinforce_agent = create_reinforce_agent(cfg, env_agents[i], i)
        acq_agents.append(acq_agent)
        reinforce_agents.append(reinforce_agent)

    tacq_agent = TemporalAgent(Agents(*acq_agents))
    tacq_agent.seed(cfg.algorithm.env_seed)

    return tacq_agent, reinforce_agents
