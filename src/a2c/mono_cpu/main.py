import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
import gym
from gym.spaces import Box, Discrete
from gym.wrappers import TimeLimit

import salina
from salina import Agent, get_arguments, get_class, instantiate_class
from salina.agents import Agents, RemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


class ProbAgent(Agent):
    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        self.set(("action_probs", t), probs)


class ActionAgent(Agent):
    def __init__(self):
        super().__init__()

    def forward(self, t, stochastic, **kwargs):
        probs = self.get(("action_probs", t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)


class CriticAgent(Agent):
    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__()
        self.critic_model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.critic_model(observation).squeeze(-1)
        self.set(("critic", t), critic)


class EnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg):
        super().__init__(
            get_class(cfg.algorithm.env),
            get_arguments(cfg.algorithm.env),
            n_envs=cfg.algorithm.n_envs,
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


class Logger:
    # Not generic
    # Specifically designed in the context of this A2C example
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string, loss, epoch):
        self.logger.add_scalar(log_string, loss.item(), epoch)

    # Log losses
    def log_losses(self, cfg, epoch, critic_loss, entropy_loss, a2c_loss):
        self.add_log("critic_loss", critic_loss, epoch)
        self.add_log("entropy_loss", entropy_loss, epoch)
        self.add_log("a2c_loss", a2c_loss, epoch)


def make_env(env_name, max_episode_steps):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


# Create the A2C gent
def create_a2c_agent(cfg, env_agent):
    observation_size, n_actions = env_agent.get_obs_and_actions_sizes()
    del env_agent.env
    prob_agent = ProbAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions
    )
    action_agent = ActionAgent()
    critic_agent = CriticAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions
    )
    # Combine env and policy agents
    agent = Agents(env_agent, prob_agent, action_agent)
    # Get an agent that is executed on a complete workspace
    agent = TemporalAgent(agent)
    agent.seed(cfg.algorithm.env_seed)
    return agent, prob_agent, critic_agent


# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, prob_agent, critic_agent):
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    parameters = nn.Sequential(prob_agent, critic_agent).parameters()
    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
    return optimizer


# ---- TRICKY PART ---- #
def execute_agent(cfg, epoch, workspace, agent):
    workspace.zero_grad()
    if epoch > 0:
        workspace.copy_n_last_steps(1)
        agent(workspace, t=1, n_steps=cfg.algorithm.n_timesteps - 1, stochastic=True)
    else:
        agent(workspace, t=0, n_steps=cfg.algorithm.n_timesteps, stochastic=True)


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


def run_a2c(cfg):
    start = time.process_time()
    # 1) Build the logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    env_agent = EnvAgent(cfg)

    # 3) Create the A2C Agent
    a2c_agent, prob_agent, critic_agent = create_a2c_agent(cfg, env_agent)

    # 4) Create the temporal critic agent to compute critic values over the workspace
    tcritic_agent = TemporalAgent(critic_agent)

    # 5) Configure the workspace to the right dimension
    workspace = salina.Workspace()

    # 6) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, prob_agent, critic_agent)

    # 7) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        execute_agent(cfg, epoch, workspace, a2c_agent)

        # Compute the critic value over the whole workspace
        tcritic_agent(workspace, n_steps=cfg.algorithm.n_timesteps)

        # Get relevant tensors (size are timestep * n_envs * ...)
        critic, done, action_probs, reward, action = workspace[
            "critic", "env/done", "action_probs", "env/reward", "action"
        ]

        # Compute critic loss
        critic_loss, td = compute_critic_loss(cfg, reward, done, critic)

        # Compute entropy loss
        entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()

        # Compute A2C loss
        a2c_loss = compute_a2c_loss(action_probs, action, td)

        # Store the losses for tensorboard display
        # logger.log_losses(cfg, epoch, critic_loss, entropy_loss, a2c_loss)

        # Compute the total loss
        loss = (
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the cumulated reward on final_state
        creward = workspace["env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_log("reward", creward.mean(), epoch)

        if creward.mean() >= 100.0:
            return epoch, time.process_time() - start


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    epoch, duration = run_a2c(cfg)

    print(f"terminated in {duration}s at epoch {epoch}")


if __name__ == "__main__":
    main()
