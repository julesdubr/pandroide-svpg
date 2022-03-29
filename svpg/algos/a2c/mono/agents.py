from salina import Agent, get_arguments, get_class, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent

import torch
import torch.nn as nn

import gym
from gym.wrappers import TimeLimit
from gym.spaces import Box, Discrete


class ProbAgent(Agent):
    """
    ProbAgent:
    - A one hidden layer neural network which takes an observation as input and whose
    output is a probability given by a final softmax layer
    - Note that to get the input observation from the environment we call:
        observation = self.get(("env/env_obs", t))
    and that to perform an action in the environment we call:
        self.set(("action_probs", t), probs)
    """

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
    """
    ActionAgent:
    - Takes action probabilities as input (coming from the ProbAgent) and outputs an
      action.
    - In the deterministic case it takes the argmax, in the stochastic case it samples
      from the Categorical distribution.
    """

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
    """
    CriticAgent:
    - A one hidden layer neural network which takes an observation as input and whose output is the value of this observation.
    - It thus implements a  V(s)  function
    """

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


def make_env(env_name, max_episode_steps):
    """
    Create the environment using gym:
    - Using hydra to take arguments from a configuration file
    """
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


class EnvAgent(AutoResetGymAgent):
    """
    Create the environment agent
    This agent implements N gym environments with auto-reset
    """

    def __init__(self, cfg, pid):
        super().__init__(
            get_class(cfg.algorithm.env),
            get_arguments(cfg.algorithm.env),
            n_envs=cfg.algorithm.n_envs,
            input=f"action{pid}",
            output=f"env{pid}/",
        )
        self.env = instantiate_class(cfg.algorithm.env)

    # This is necessary to create the corresponding RL agent
    def get_obs_and_actions_sizes(self):
        if isinstance(self.env.action_space, Box):
            # Return the size of the observation and action spaces of the environment
            # In the case of a continuous action environment
            return self.env.observation_space.shape[0], self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, Discrete):
            # Return the size of the observation and action spaces of the environment
            return self.env.observation_space.shape[0], self.env.action_space.n
        else:
            print("unknown type of action space", self.env.action_space)
            return None


def create_a2c_agent(cfg, env_agent, pid):
    # Get info on the environment
    observation_size, n_actions = env_agent.get_obs_and_actions_sizes()
    del env_agent.env

    env_agent = EnvAgent(cfg, pid)

    prob_agent = ProbAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions, pid
    )

    action_agent = ActionAgent(pid)

    # Combine env and acquisition agents
    # We'll combine the acq_agents of all particle into a single TemporalAgent later
    acq_agent = Agents(env_agent, prob_agent, action_agent)

    critic_agent = CriticAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, pid
    )

    return acq_agent, prob_agent, critic_agent


def combine_agents(cfg, acq_agents, critic_agents):
    # Combine all acquisition agent of all particle in a unique TemporalAgent.
    # This will help us to avoid using a loop explicitly to execute all these agents
    # (these agents will still be executed by a for loop by SaliNa)
    tacq_agent = TemporalAgent(Agents(*acq_agents))

    # Set the seed
    tacq_agent.seed(cfg.algorithm.env_seed)

    # We also combine all the critic_agent of all particle into a unique TemporalAgent
    tcritic_agent = TemporalAgent(Agents(*critic_agents))

    return tacq_agent, tcritic_agent


def create_particles(cfg, n_particles, env_agents):
    acq_agents = list()
    prob_agents = list()
    critic_agents = list()

    for i in range(n_particles):
        # Create A2C agent for all particles
        acq_agent, prob_agent, critic_agent = create_a2c_agent(cfg, env_agents[i], i)
        acq_agents.append(acq_agent)
        prob_agents.append(prob_agent)
        critic_agents.append(critic_agent)

    return acq_agents, prob_agents, critic_agents


def execute_agent(cfg, epoch, workspace, agent):
    """
    Execute agent:
    - This is the tricky part with SaLinA, the one we need to understand in detail.
    - The difficulty lies in the copy of the last step and the way to deal with the
      n_steps return.
    - The call to agent(workspace, t=1, n_steps=cfg.algorithm.n_timesteps - 1,
      stochastic=True) makes the agent run a number of steps in the workspace.
      n practice, it calls makes a forward pass of the agent network using the
      workspace data and updates the workspace accordingly.
    - Now, if we start at the first epoch (epoch=0), we start from the first step (t=0).
      But when subsequently we perform the next epochs (epoch>0), there is a risk that
      we do not cover the transition at the border between the previous epoch and the
      current epoch. To avoid this risk, we need to shift the time indexes, hence the
      (t=1) and (cfg.algorithm.n_timesteps - 1).
    """
    if epoch > 0:
        workspace.zero_grad()
        workspace.copy_n_last_steps(1)
        agent(workspace, t=1, n_steps=cfg.algorithm.n_timesteps - 1, stochastic=True)
    else:
        agent(workspace, t=0, n_steps=cfg.algorithm.n_timesteps, stochastic=True)
