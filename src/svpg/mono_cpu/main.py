import copy
import time

import numpy as np

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
    def __init__(self, observation_size, hidden_size, n_actions, pid):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )
        self.pid = pid

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        scores = self.model(observation)
        probs = torch.softmax(scores, dim=-1)
        self.set(("action_probs" + str(self.pid), t), probs)


class ActionAgent(Agent):
    def __init__(self, pid):
        super().__init__()
        self.pid = pid

    def forward(self, t, stochastic, **kwargs):
        probs = self.get(("action_probs" + str(self.pid), t))
        if stochastic:
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = probs.argmax(1)

        self.set(("action", t), action)


class CriticAgent(Agent):
    def __init__(self, observation_size, hidden_size, n_actions, pid):
        super().__init__()
        self.critic_model = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.pid = pid

    def forward(self, t, **kwargs):
        observation = self.get(("env/env_obs", t))
        critic = self.critic_model(observation).squeeze(-1)
        self.set(("critic" + str(self.pid), t), critic)


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


class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


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
def create_a2c_agent(cfg, env_agent, pid, n_particles):
    observation_size, n_actions = env_agent.get_obs_and_actions_sizes()
    del env_agent.env

    prob_agent = ProbAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions, pid
    )

    action_agent = ActionAgent(pid)
    critic_agent = CriticAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions, pid
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


def get_parameters(nn_list):
    params = []
    for nn in nn_list:
        l = list(nn.parameters())
        # l_flatten = (torch.flatten(p) for p in l)

        l_flatten = []

        for p in l:
            l_flatten.append(torch.flatten(p))

        l_flatten = tuple(l_flatten)
        l_concat = torch.cat(l_flatten)

        params.append(l_concat)

    return torch.stack(params)


def add_gradients(total_a2c_loss, kernels, particles, n_particles, temp):
    total_a2c_loss.backward(retain_graph=True)

    for i in range(n_particles):
        for j in range(n_particles):
            if i == j:
                continue

            theta_i = particles[i]["prob_agent"].model.parameters()
            theta_j = particles[j]["prob_agent"].model.parameters()

            for (wi, wj) in zip(theta_i, theta_j):
                wi.grad = wi.grad + wj.grad * kernels[j, i].detach()


def add_gradient_to_nn_params(nn, grad, n_particles):
    for i, theta in enumerate(nn.parameters()):
        if theta.grad is None:
            theta.grad = grad[i] / n_particles

        theta.grad = theta.grad + grad[i] / n_particles


def run_svpg(cfg, n_particles=16, temp=1):
    start = time.process_time()

    # 1) Build the logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    env_agent = EnvAgent(cfg)

    # 3) Create the A2C Agent
    # Store all differents particles in a dictionary
    particles = dict()
    for i in range(n_particles):
        a2c_agent, prob_agent, critic_agent = create_a2c_agent(
            cfg, env_agent, i, n_particles
        )
        particles[i] = {
            "a2c_agent": a2c_agent,
            "prob_agent": prob_agent,
            "critic_agent": critic_agent,
        }

    # 4) Create the temporal critic agent to compute critic values over the workspace
    tcritic_agents = [
        TemporalAgent(particles[i]["critic_agent"]) for i in range(n_particles)
    ]

    # 5) Configure the workspace to the right dimension
    workspace = salina.Workspace()

    # 6) Configure the optimizer over the a2c agent
    optimizers = [
        setup_optimizers(cfg, particles[i]["prob_agent"], particles[i]["critic_agent"])
        for i in range(n_particles)
    ]

    # 7) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):
        total_critic_loss = None
        total_entropy_loss = None
        total_a2c_loss = None

        for i in range(n_particles):
            # Execute the agent in the workspace
            execute_agent(cfg, epoch, workspace, particles[i]["a2c_agent"])

            # Compute the critic value over the whole workspace
            tcritic_agents[i](workspace, n_steps=cfg.algorithm.n_timesteps)

            # Get relevant tensors (size are timestep * n_envs * ...)
            critic, done, action_probs, reward, action = workspace[
                "critic" + str(i),
                "env/done",
                "action_probs" + str(i),
                "env/reward",
                "action",
            ]

            # Compute critic loss
            critic_loss, td = compute_critic_loss(cfg, reward, done, critic)
            if total_critic_loss is None:
                total_critic_loss = critic_loss
            else:
                total_critic_loss = total_critic_loss + critic_loss

            # Compute entropy loss
            entropy_loss = (
                torch.distributions.Categorical(action_probs).entropy().mean()
            )
            if total_entropy_loss is None:
                total_entropy_loss = entropy_loss
            else:
                total_entropy_loss = total_entropy_loss + entropy_loss

            # Compute A2C loss
            a2c_loss = compute_a2c_loss(action_probs, action, td)
            if total_a2c_loss is None:
                total_a2c_loss = -a2c_loss
            else:
                total_a2c_loss = total_a2c_loss - a2c_loss * (1 / temp) * (
                    1 / n_particles
                )

        params = get_parameters(
            [particles[i]["prob_agent"].model for i in range(n_particles)]
        )

        kernels = RBF()(params, params.detach())

        add_gradients(total_a2c_loss, kernels, particles, n_particles, temp)

        # Store the losses for tensorboard display
        # logger.log_losses(cfg, epoch, critic_loss, entropy_loss, a2c_loss)

        # Compute the total loss
        loss = (
            -cfg.algorithm.entropy_coef * total_entropy_loss
            + cfg.algorithm.critic_coef * total_critic_loss
            - cfg.algorithm.a2c_coef * total_a2c_loss
            - kernels.sum()
        )

        for i in range(n_particles):
            optimizers[i].zero_grad()

        loss.backward()

        for i in range(n_particles):
            optimizers[i].step()

        # Compute the cumulated reward on final_state
        creward = workspace["env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_log("reward", creward.mean(), epoch)

        if creward.mean() >= 100.0:
            break

    return epoch, time.process_time() - start


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    epoch, duration = run_svpg(cfg)

    print(f"terminated in {duration}s at epoch {epoch}")


if __name__ == "__main__":
    main()
