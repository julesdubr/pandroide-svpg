from copy import deepcopy
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
from salina import Agent, Workspace, get_arguments, get_class, instantiate_class
from salina.agents import Agents, NRemoteAgent, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent, GymAgent
from salina.logger import TFLogger

from visu.visu_gradient import visu_loss_along_time


def _index(tensor_3d, tensor_2d):
    """This function is used to index a 3d tensors using a 2d tensor"""
    x, y, z = tensor_3d.size()
    t = tensor_3d.reshape(x * y, z)
    tt = tensor_2d.reshape(x * y)
    v = t[torch.arange(x * y), tt]
    v = v.reshape(x, y)
    return v


class Logger:
    # Not generic, specifically designed in the context of this A2C example
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string, loss, epoch):
        self.logger.add_scalar(log_string, loss.item(), epoch)

    # Log losses
    def log_losses(self, cfg, epoch, critic_loss, entropy_loss, a2c_loss):
        self.add_log("critic_loss", critic_loss, epoch)
        self.add_log("entropy_loss", entropy_loss, epoch)
        self.add_log("a2c_loss", a2c_loss, epoch)


class REINFORCEAgent(Agent):
    def __init__(self, observation_size, hidden_size, n_actions, pid):
        super().__init__(name=f"r_agent{pid}")
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

        gamma = 1.0 / (2 * sigma ** 2) if sigma != 0 else 1e8

        K_XY = (-gamma * dnorm2).exp()

        return K_XY


# TODO: tester environement custom ?
def make_env(env_name, max_episode_steps):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


# Create the A2C gent
def create_reinforce_agent(cfg, env_agent, pid):
    # Get info on the environment
    observation_size, n_actions = env_agent.get_obs_and_actions_sizes()
    del env_agent.env

    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0

    acq_env_agent = EnvAgent(cfg, pid)

    r_agent = REINFORCEAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions, pid
    )
    acq_r_agent = deepcopy(r_agent)

    acq_agent = Agents(acq_env_agent, acq_r_agent)

    return acq_agent, r_agent


def combine_agents(cfg, particles):
    # Combine all acquisition agent of all particle in a unique TemporalAgent.
    # This will help us to avoid using a loop explicitly to execute all these agents
    # (these agents will still be executed by a for loop by SaliNa)
    acq_agents = TemporalAgent(
        Agents(*[particle["acq_agent"] for particle in particles])
    )

    # Create the remote acquisition agent and the remote acquisition workspace
    acq_remote_agents, acq_workspace = NRemoteAgent.create(
        acq_agents,
        num_processes=cfg.algorithm.n_processes,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        stochastic=True,
    )
    # Set the seed
    acq_remote_agents.seed(cfg.algorithm.env_seed)

    # Combine all prob_agent of each particle to calculate the gradient
    r_agents = Agents(*[particle["r_agent"] for particle in particles])

    return r_agents, acq_remote_agents, acq_workspace


def create_particles(cfg, n_particles, env_agents):
    particles = list()
    for i in range(n_particles):
        # Create A2C agent for all particles
        acq_agent, r_agent = create_reinforce_agent(cfg, env_agents[i], i)
        particles.append(
            {
                "acq_agent": acq_agent,
                "r_agent": r_agent,
            }
        )

    return particles


# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, r_agents):
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    # parameters = [r_agent.parameters() for r_agent in r_agents]
    # parameters = nn.Sequential(**r_agents).parameters()

    parameters = []
    for r_agent in r_agents:
        parameters += list(r_agent.parameters())

    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
    return optimizer


def execute_agent(cfg, epoch, acq_remote_agents, acq_workspace, particles):
    for i, particle in enumerate(particles):
        pagent = acq_remote_agents.get_by_name(f"r_agent{i}")
        for a in pagent:
            a.load_state_dict(particle["r_agent"].state_dict())

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


def compute_reinforce_loss(
    reward, action_probabilities, baseline, action, done, discount_factor
):
    """This function computes the reinforce loss, considering that episodes may have
    different lengths."""
    batch_size = reward.size()[1]

    # Find the first done occurence for each episode
    v_done, trajectories_length = done.float().max(0)
    trajectories_length += 1
    print(v_done)
    # assert v_done.eq(1.0).all()
    max_trajectories_length = trajectories_length.max().item()

    # Shorten trajectories for accelerate computation
    reward = reward[:max_trajectories_length]
    action_probabilities = action_probabilities[:max_trajectories_length]
    baseline = baseline[:max_trajectories_length]
    action = action[:max_trajectories_length]

    # Create a binary mask to mask useless values (of size max_trajectories_length x batch_size)
    arange = (
        torch.arange(max_trajectories_length, device=done.device)
        .unsqueeze(-1)
        .repeat(1, batch_size)
    )
    mask = arange.lt(
        trajectories_length.unsqueeze(0).repeat(max_trajectories_length, 1)
    )
    reward = reward * mask

    # Compute discounted cumulated reward
    cumulated_reward = [torch.zeros_like(reward[-1])]
    for t in range(max_trajectories_length - 1, 0, -1):
        cumulated_reward.append(discount_factor + cumulated_reward[-1] + reward[t])
    cumulated_reward.reverse()
    cumulated_reward = torch.cat([c.unsqueeze(0) for c in cumulated_reward])

    # baseline loss
    g = baseline - cumulated_reward
    baseline_loss = (g) ** 2
    baseline_loss = (baseline_loss * mask).mean()

    # policy loss
    log_probabilities = _index(action_probabilities, action).log()
    policy_loss = log_probabilities * -g.detach()
    policy_loss = policy_loss * mask
    policy_loss = policy_loss.mean()

    # entropy loss
    entropy = torch.distributions.Categorical(action_probabilities).entropy() * mask
    entropy_loss = entropy.mean()

    return {
        "baseline_loss": baseline_loss,
        "reinforce_loss": policy_loss,
        "entropy_loss": entropy_loss,
    }


def compute_total_loss(cfg, particles, replay_workspace, alpha, logger, epoch, verbose):
    n_particles = len(particles)
    stop = False

    # Compute critic, entropy and a2c losses
    reinforce_loss, entropy_loss, baseline_loss = 0, 0, 0
    for i in range(n_particles):
        # Get relevant tensors (size are timestep * n_envs * ...)
        baseline, done, action_probs, reward, action = replay_workspace[
            f"baseline{i}",
            f"env{i}/done",
            f"action_probs{i}",
            f"env{i}/reward",
            f"action{i}",
        ]

        # Compute critic loss
        losses = compute_reinforce_loss(
            reward, action_probs, baseline, action, done, cfg.algorithm.discount_factor
        )

        if verbose:
            [logger.add_scalar(k, v.item(), epoch) for k, v in losses.items()]

        entropy_loss += losses["entropy_loss"]
        baseline_loss += losses["baseline_loss"]
        reinforce_loss -= losses["reinforce_loss"]  # * (1 / alpha) * (1 / n_particles)

        # Compute the cumulated reward on final_state
        creward = replay_workspace[f"env{i}/cumulated_reward"]
        tl = done.float().argmax(0)
        creward = creward[tl, torch.arange(creward.size()[1])]

        if creward.size()[0] > 0:
            logger.add_log(f"reward{i}", creward.mean(), epoch)

        if creward.mean() >= 100:
            stop = True

    # Get the params
    params = get_parameters([particles[i]["r_agent"].model for i in range(n_particles)])

    # We need to detach the second list of params out of the computation graph
    # because we don't want to compute its gradient two time when using backward()
    kernels = RBF()(params, params.detach())

    # Compute the first term in the SVGD update
    add_gradients(reinforce_loss, kernels, particles, n_particles)

    loss = (
        -cfg.algorithm.entropy_coef * entropy_loss
        + cfg.algorithm.baseline_coef * baseline_loss
        + cfg.algorithm.reinforce_coef * reinforce_loss
        # + kernels.sum() / n_particles
    )

    return loss, stop


def compute_gradients_norms(particles, logger, epoch):
    policy_gradnorm = 0

    for particle in particles:

        r_params = particle["r_agent"].model.parameters()

        for w in r_params:
            if w.grad != None:
                policy_gradnorm += w.grad.detach().data.norm(2) ** 2

    policy_gradnorm = torch.sqrt(policy_gradnorm)

    logger.add_log("Policy Gradient norm", policy_gradnorm, epoch)


def get_parameters(nn_list):
    params = []
    for nn in nn_list:
        l = list(nn.parameters())

        l_flatten = [torch.flatten(p) for p in l]
        l_flatten = tuple(l_flatten)

        l_concat = torch.cat(l_flatten)

        params.append(l_concat)

    return torch.stack(params)


def add_gradients(total_a2c_loss, kernels, particles, n_particles):
    total_a2c_loss.backward(retain_graph=True)

    for i in range(n_particles):
        for j in range(n_particles):
            if i == j:
                continue

            theta_i = particles[i]["r_agent"].model.parameters()
            theta_j = particles[j]["r_agent"].model.parameters()

            for (wi, wj) in zip(theta_i, theta_j):
                wi.grad = wi.grad + wj.grad * kernels[j, i].detach()


def run_svpg(cfg, alpha=1, show_losses=False, show_gradients=False):
    losses = []

    # 1) Build the logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    n_particles = cfg.algorithm.n_particles
    env_agents = [EnvAgent(cfg, i) for i in range(n_particles)]

    # 3) Create the particles
    particles = create_particles(cfg, n_particles, env_agents)

    # 4) Combine the agents
    r_agents, acq_remote_agents, acq_workspace = combine_agents(cfg, particles)

    # 5) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, [particle["r_agent"] for particle in particles])

    # 8) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the remote acq_agent in the remote workspace
        execute_agent(cfg, epoch, acq_remote_agents, acq_workspace, particles)

        # Compute the prob and critic value over the whole replay workspace
        replay_workspace = Workspace(acq_workspace)

        for i, agent in enumerate(r_agents.agents):
            agent(replay_workspace, stochastic=True, t=0, stop_variable=f"env{i}/done")

        # Sum up all the losses including the sum of kernel matrix and then use
        # backward() to automatically compute the gradient of the critic and the
        # second term in SVGD update
        loss, stop = compute_total_loss(
            cfg, particles, replay_workspace, alpha, logger, epoch, show_losses
        )
        losses.append(loss.item())

        if stop:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the norm of gradient of the actor and gradient of the critic
        if show_gradients:
            compute_gradients_norms(particles, logger, epoch)

    return losses, epoch


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    duration = time.process_time()
    losses, epoch = run_svpg(cfg)
    duration = time.process_time() - duration

    visu_loss_along_time(range(epoch + 1), losses, "loss_along_time")

    print(f"terminated in {duration}s at epoch {epoch}")


if __name__ == "__main__":
    main()
