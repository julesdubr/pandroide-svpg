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
        observation = self.get(("env" + str(self.pid) + "/env_obs", t))
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

        self.set(("action" + str(self.pid), t), action)


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
        observation = self.get(("env" + str(self.pid) + "/env_obs", t))
        critic = self.critic_model(observation).squeeze(-1)
        self.set(("critic" + str(self.pid), t), critic)


class EnvAgent(AutoResetGymAgent):
    # Create the environment agent
    # This agent implements N gym environments with auto-reset
    def __init__(self, cfg, pid):
        super().__init__(
            get_class(cfg.algorithm.env),
            get_arguments(cfg.algorithm.env),
            n_envs=int(cfg.algorithm.n_envs / cfg.algorithm.n_processes),
            # add the pid of the agent corresponding to the environment to the input
            # and output
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

        if sigma != 0:
            gamma = 1.0 / (2 * sigma ** 2)
        else:
            gamma = 1e8

        K_XY = (-gamma * dnorm2).exp()

        return K_XY


def make_env(env_name, max_episode_steps):
    return TimeLimit(gym.make(env_name), max_episode_steps=max_episode_steps)


# Create the A2C gent
def create_a2c_agent(cfg, env_agent, pid):
    observation_size, n_actions = env_agent.get_obs_and_actions_sizes()
    del env_agent.env

    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0

    prob_agent = ProbAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions, pid
    )
    acq_prob_agent = deepcopy(prob_agent)  # create a copy of the prob_agent

    action_agent = ActionAgent(pid)

    # Combine env and acquisition agents
    # We'll combine the acq_agents of all particle into a single TemporalAgent later
    acq_agent = Agents(env_agent, acq_prob_agent, action_agent)

    critic_agent = CriticAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions, pid
    )

    return acq_agent, prob_agent, critic_agent


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
    tprob_agents = Agents(*[particle["prob_agent"] for particle in particles])

    # We also combine all the critic_agent of all particle into a unique TemporalAgent
    tcritic_agent = TemporalAgent(
        Agents(*[particle["critic_agent"] for particle in particles])
    )

    return tprob_agents, tcritic_agent, acq_remote_agents, acq_workspace


def create_particles(cfg, n_particles, env_agents):
    particles = list()
    for i in range(n_particles):
        # Create A2C agent for all particles
        acq_agent, prob_agent, critic_agent = create_a2c_agent(cfg, env_agents[i], i)
        particles.append(
            {
                "acq_agent": acq_agent,
                "prob_agent": prob_agent,
                "critic_agent": critic_agent,
            }
        )

    return particles


# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, prob_agents, critic_agents):
    optimizer_args = get_arguments(cfg.algorithm.optimizer)

    parameters = []

    for nn in zip(prob_agents, critic_agents):
        parameters = parameters + list(nn[0].parameters()) + list(nn[1].parameters())

    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
    return optimizer


def execute_agent(cfg, epoch, acq_remote_agents, acq_remote_workspace, particles):
    for i, particle in enumerate(particles):
        for a in acq_remote_agents.get_by_name(f"prob_agent{i}"):
            a.load_state_dict(particle["prob_agent"].state_dict())

    if epoch > 0:
        acq_remote_workspace.zero_grad()
        acq_remote_workspace.copy_n_last_steps(1)
        acq_remote_agents(
            acq_remote_workspace,
            t=1,
            n_steps=cfg.algorithm.n_timesteps - 1,
            stochastic=True,
        )
    else:
        acq_remote_agents(
            acq_remote_workspace,
            t=0,
            n_steps=cfg.algorithm.n_timesteps,
            stochastic=True,
        )


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


def compute_losses(cfg, n_particles, replay_workspace, alpha, logger, epoch):
    critic_loss, entropy_loss, a2c_loss = 0, 0, 0

    for i in range(n_particles):
        # Get relevant tensors (size are timestep * n_envs * ...)
        critic, done, action_probs, reward, action = replay_workspace[
            f"critic{i}",
            f"env{i}/done",
            f"action_probs{i}",
            f"env{i}/reward",
            f"action{i}",
        ]

        # Compute critic loss
        tmp, td = compute_critic_loss(cfg, reward, done, critic)
        critic_loss += tmp

        # Compute entropy loss
        entropy_loss += torch.distributions.Categorical(action_probs).entropy().mean()

        # Compute A2C loss
        a2c_loss -= (
            compute_a2c_loss(action_probs, action, td) * (1 / alpha) * (1 / n_particles)
        )

        # Compute the cumulated reward on final_state
        creward = replay_workspace[f"env{i}/cumulated_reward"]
        creward = creward[done]

        # if creward.size()[0] > 0:
        #     logger.add_log(f"reward{i}", creward.mean(), epoch)

    return critic_loss, entropy_loss, a2c_loss


def compute_gradients_norms(particles, logger, epoch):
    policy_gradnorm, critic_gradnorm = 0, 0

    for particle in particles:

        prob_params = particle["prob_agent"].model.parameters()
        critic_params = particle["critic_agent"].critic_model.parameters()

        for w_prob, w_critic in zip(prob_params, critic_params):
            if w_prob.grad != None:
                policy_gradnorm += w_prob.grad.detach().data.norm(2) ** 2

            if w_critic.grad != None:
                critic_gradnorm += w_critic.grad.detach().data.norm(2) ** 2

    policy_gradnorm, critic_gradnorm = (
        torch.sqrt(policy_gradnorm),
        torch.sqrt(critic_gradnorm),
    )

    logger.add_log("Policy Gradient norm", policy_gradnorm, epoch)
    logger.add_log("Critic Gradient norm", critic_gradnorm, epoch)


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

            theta_i = particles[i]["prob_agent"].model.parameters()
            theta_j = particles[j]["prob_agent"].model.parameters()

            for (wi, wj) in zip(theta_i, theta_j):
                wi.grad = wi.grad + wj.grad * kernels[j, i].detach()


def run_svpg(cfg, alpha=1):
    # 1) Build the logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    n_particles = cfg.algorithm.n_particles
    env_agents = [EnvAgent(cfg, i) for i in range(n_particles)]

    # 3) Create the particles
    particles = create_particles(cfg, n_particles, env_agents)

    # 4) Combine the agents
    tprob_agents, tcritic_agent, acq_remote_agents, acq_workspace = combine_agents(
        cfg, particles
    )

    # 5) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(
        cfg,
        [particle["prob_agent"] for particle in particles],
        [particle["critic_agent"] for particle in particles],
    )

    # 8) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Zero the gradient
        optimizer.zero_grad()

        # Execute the remote acq_agent in the remote workspace
        execute_agent(cfg, epoch, acq_remote_agents, acq_workspace, particles)

        # Compute the prob and critic value over the whole replay workspace
        replay_workspace = Workspace(acq_workspace)
        tprob_agents(replay_workspace, t=0, n_steps=cfg.algorithm.n_timesteps)
        tcritic_agent(replay_workspace, t=0, n_steps=cfg.algorithm.n_timesteps)

        # Compute the losses
        critic_loss, entropy_loss, a2c_loss = compute_losses(
            cfg, n_particles, replay_workspace, alpha, logger, epoch
        )

        params = get_parameters(
            [particles[i]["prob_agent"].model for i in range(n_particles)]
        )

        # We need to detach the second list of params out of the computation graph
        # because we don't want to compute its gradient two time when using backward()
        kernels = RBF()(params, params.detach())

        # Compute the first term in the SVGD update
        add_gradients(a2c_loss, kernels, particles, n_particles)

        # Sum up all the loss including the sum of kernel matrix and then use backward()
        # to automatically compute the gradient of the critic and the second term in
        # SVGD update
        total_loss = (
            cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.entropy_coef * entropy_loss
            + kernels.sum() / n_particles
        )

        total_loss.backward()
        optimizer.step()

        # Compute the norm of gradient of the actor and gradient of the critic
        compute_gradients_norms(particles, logger, epoch)

        # Store the mean of losses all over the agents for tensorboard display
        logger.log_losses(
            cfg,
            epoch,
            critic_loss.detach().mean(),
            entropy_loss.detach().mean(),
            a2c_loss.detach().mean(),
        )

    return epoch


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    duration = time.process_time()
    epoch = run_svpg(cfg)
    duration = time.process_time() - duration

    print(f"terminated in {duration}s at epoch {epoch}")


if __name__ == "__main__":
    main()
