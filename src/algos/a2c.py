from copy import deepcopy

import torch
import torch.nn as nn

import gym
from gym.spaces import Box, Discrete
from gym.wrappers import TimeLimit

from salina import Agent, get_arguments, get_class, instantiate_class
from salina.agents import Agents
from salina.agents.gyma import AutoResetGymAgent


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


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
            n_envs=int(cfg.algorithm.n_envs / cfg.algorithm.n_processes),
            input=f"action{pid}",
            output=f"env{pid}/",
        )
        self.env = instantiate_class(cfg.algorithm.env)

    # Return the size of the observation and action spaces of the env
    # TODO: fix Pendulum
    def get_obs_and_actions_sizes(self):
        action_space = None

        if isinstance(self.env.action_space, Box):
            action_space = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, Discrete):
            action_space = self.env.action_space.n

        return self.env.observation_space.shape[0], action_space


def make_env(env_name, max_episode_steps):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


# Create the A2C gent
def create_a2c_agent(cfg, env_agent, pid):
    # Get info on the environment
    observation_size, n_actions = env_agent.get_obs_and_actions_sizes()
    del env_agent.env

    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0

    acq_env_agent = EnvAgent(cfg, pid)

    prob_agent = ProbAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions, pid
    )
    acq_prob_agent = deepcopy(prob_agent)  # create a copy of the prob_agent

    action_agent = ActionAgent(pid)

    # Combine env and acquisition agents
    # We'll combine the acq_agents of all particle into a single TemporalAgent later
    acq_agent = Agents(acq_env_agent, acq_prob_agent, action_agent)

    critic_agent = CriticAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, pid
    )

    return acq_agent, prob_agent, critic_agent


def compute_critic_loss(cfg, reward, done, critic):
    # Compute de temporal difference
    target = reward[1:] + cfg.algorithm.discount_factor * critic[1:].detach() * (
        1 - done[1:].float()
    )
    td = target - critic[:-1]

    # Compute critic loss
    td_error = td ** 2
    critic_loss = td_error.mean()
    return critic_loss, td


def compute_a2c_loss(action_probs, action, td):
    action_logp = _index(action_probs, action).log()
    a2c_loss = action_logp[:-1] * td.detach()
    return a2c_loss.mean()


# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, prob_agents, critic_agents):
    optimizer_args = get_arguments(cfg.algorithm.optimizer)

    parameters = []
    for prob_agent, critic_agent in zip(prob_agents, critic_agents):
        parameters += list(prob_agent.parameters()) + list(critic_agent.parameters())

    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
    return optimizer


def execute_agent(cfg, epoch, acq_remote_agents, acq_workspace, particles):
    for i, particle in enumerate(particles):
        pagent = acq_remote_agents.get_by_name(f"prob_agent{i}")
        for a in pagent:
            a.load_state_dict(particle["prob_agent"].state_dict())

    if epoch > 0:
        # acq_remote_workspace.zero_grad()
        acq_workspace.copy_n_last_steps(1)
        acq_remote_agents(
            acq_workspace, t=1, n_steps=cfg.algorithm.n_timesteps - 1, stochastic=True
        )
    else:
        acq_remote_agents(
            acq_workspace, t=0, n_steps=cfg.algorithm.n_timesteps, stochastic=True
        )
