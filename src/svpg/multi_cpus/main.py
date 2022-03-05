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
from salina import Agent, get_arguments, get_class, instantiate_class
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
            n_envs=cfg.algorithm.n_envs,
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

    prob_agent = ProbAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions, pid
    )

    # We need to add the pid of the particle to its prob_agent name so
    # that we can synchronize the acquisition_agent of each particle to
    # the prob_agent corresponding
    prob_agent.set_name("prob_agent" + str(pid))

    # create a copy of the prob_agent
    acquisition_prob_agent = deepcopy(prob_agent)

    action_agent = ActionAgent(pid)

    critic_agent = CriticAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions, pid
    )

    # Combine env and acquisition agents
    # We will combine all the acquisition_agent of all particle into a TemporalAgent later
    acquisition_agent = Agents(env_agent, acquisition_prob_agent, action_agent)

    return acquisition_agent, prob_agent, critic_agent


# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, prob_agents, critic_agents):
    optimizer_args = get_arguments(cfg.algorithm.optimizer)

    parameters = []

    for nn in zip(prob_agents, critic_agents):
        parameters = parameters + list(nn[0].parameters()) + list(nn[1].parameters())

    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
    return optimizer


def execute_agent(cfg, epoch, workspace, agent):
    if epoch > 0:
        workspace.zero_grad()
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


def run_svpg(cfg, temp=1):
    start = time.process_time()

    # 1) Build the logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    n_particles = cfg.algorithm.n_particles
    env_agents = [EnvAgent(cfg, i) for i in range(n_particles)]

    # 3) Create the A2C Agent
    # Store all differents particles in a dictionary
    particles = list()
    for i in range(n_particles):
        acquisition_agent, prob_agent, critic_agent = create_a2c_agent(
            cfg, env_agents[i], i
        )
        particle = {
            "acquisition_agent": acquisition_agent,
            "prob_agent": prob_agent,
            "critic_agent": critic_agent,
        }
        particles.append(particle)

    # Combine all acquisition agent of all particle in a unique TemporalAgent. This will help us to avoid
    # using a loop explicitly to execute all these agents (these agents will still be executed by a for loop by SaliNa)
    combined_acquisition_agent = TemporalAgent(
        Agents(*[particle["acquisition_agent"] for particle in particles])
    )

    # Combine all prob_agent of each particle to calculate the gradient
    combined_prob_agent = Agents(*[particle["prob_agent"] for particle in particles])

    # Create the remote acquisition agent and the remote acquisition workspace
    remote_combined_acq_agent, remote_acquisition_workspace = NRemoteAgent.create(
        combined_acquisition_agent,
        num_processes=cfg.algorithm.n_processes,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        stochastic=True,
    )

    # Set the seed
    remote_combined_acq_agent.seed(cfg.algorithm.env_seed)

    # 4) Create the temporal critic agent to compute critic values over the workspace
    # We also combine all the critic_agent of all particle into a unique TemporalAgent
    tcritic_agent = TemporalAgent(
        Agents(*[particle["critic_agent"] for particle in particles])
    )

    # 5) Configure the workspace to the right dimension
    workspace = salina.Workspace()

    # 6) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(
        cfg,
        [particle["prob_agent"] for particle in particles],
        [particle["critic_agent"] for particle in particles],
    )

    # 7) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Zero the gradient
        optimizer.zero_grad()
        # At each epoch, we have to synchronize, for each particle, its acquisiton_agent and its prob_agent
        for i in range(n_particles):
            for a in remote_combined_acq_agent.get_by_name("prob_agent" + str(i)):
                a.load_state_dict(particles[i]["prob_agent"].state_dict())

        # Intialte the losses
        total_critic_loss = None
        total_entropy_loss = None
        total_a2c_loss = None

        # norm of gradients for debugging
        total_policy_gradnorm = None
        total_critic_gradnorm = None

        # Execute the remote acquisition_agent in the remote workspace
        execute_agent(
            cfg, epoch, remote_acquisition_workspace, remote_combined_acq_agent
        )

        # Create a copy of workspace to replay the critic_agent and the prob_agent in order to
        # compute the gradient
        workspace = salina.Workspace(remote_acquisition_workspace)

        # Replay the prob_agent
        combined_prob_agent(
            workspace,
            t=0,
            n_steps=cfg.algorithm.n_timesteps,
            replay=True,
            stochastic=True,
        )

        # Replay the critic_agent
        tcritic_agent(workspace, n_steps=cfg.algorithm.n_timesteps, replay=True)

        for i in range(n_particles):
            # Get relevant tensors (size are timestep * n_envs * ...)
            critic, done, action_probs, reward, action = workspace[
                "critic" + str(i),
                "env" + str(i) + "/done",
                "action_probs" + str(i),
                "env" + str(i) + "/reward",
                "action" + str(i),
            ]

            # Compute critic loss
            critic_loss, td = compute_critic_loss(cfg, reward, done, critic)
            if total_critic_loss:
                total_critic_loss = total_critic_loss + critic_loss
            else:
                total_critic_loss = critic_loss

            # Compute entropy loss
            entropy_loss = (
                torch.distributions.Categorical(action_probs).entropy().mean()
            )
            if total_entropy_loss:
                total_entropy_loss = total_entropy_loss + entropy_loss
            else:
                total_entropy_loss = entropy_loss

            # Compute A2C loss
            a2c_loss = compute_a2c_loss(action_probs, action, td)
            if total_a2c_loss:
                total_a2c_loss -= a2c_loss * (1 / temp) * (1 / n_particles)
            else:
                total_a2c_loss = -a2c_loss * (1 / temp) * (1 / n_particles)

            # Compute the cumulated reward on final_state
            creward = workspace["env" + str(i) + "/cumulated_reward"]
            creward = creward[done]

            # if creward.size()[0] > 0:
            #     logger.add_log("reward" + str(i), creward.mean(), epoch)

        params = get_parameters(
            [particles[i]["prob_agent"].model for i in range(n_particles)]
        )

        # We need to detach the second list of params out of the computation graph
        # because we don't want to compute its gradient two time when using backward()
        kernels = RBF()(params, params.detach())

        # Compute the first term in the SVGD update
        add_gradients(total_a2c_loss, kernels, particles, n_particles)

        # Sum up all the loss including the sum of kernel matrix and then use backward() to automatically compute the gradient of the critic
        # and the second term in SVGD update
        total_loss = (
            cfg.algorithm.critic_coef * total_critic_loss
            - cfg.algorithm.entropy_coef * total_entropy_loss
            + kernels.sum() / n_particles
        )

        total_loss.backward()

        optimizer.step()

        # Compute the norm of gradient of the actor and gradient of the critic
        for i in range(n_particles):

            prob_params = particles[i]["prob_agent"].model.parameters()
            critic_params = particles[i]["critic_agent"].critic_model.parameters()

            for p1, p2 in zip(prob_params, critic_params):
                if p1.grad is not None:
                    norm = p1.grad.detach().data.norm(2)
                    if total_policy_gradnorm is not None:
                        total_policy_gradnorm = total_policy_gradnorm + norm ** 2
                    else:
                        total_policy_gradnorm = norm ** 2

                if p2.grad is not None:
                    norm = p2.grad.detach().data.norm(2)
                    if total_critic_gradnorm is not None:
                        total_critic_gradnorm = total_critic_gradnorm + norm ** 2
                    else:
                        total_critic_gradnorm = norm ** 2

        total_policy_gradnorm, total_critic_gradnorm = (
            total_policy_gradnorm ** 0.5,
            total_critic_gradnorm ** 0.5,
        )
        logger.add_log("Policy Gradient norm", total_policy_gradnorm, epoch)
        logger.add_log("Critic Gradient norm", total_critic_gradnorm, epoch)

        # Store the mean of losses all over the agents for tensorboard display
        logger.log_losses(
            cfg,
            epoch,
            total_critic_loss.detach().mean(),
            total_entropy_loss.detach().mean(),
            total_a2c_loss.detach().mean(),
        )

    return epoch, time.process_time() - start


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    epoch, duration = run_svpg(cfg)

    print(f"terminated in {duration}s at epoch {epoch}")


if __name__ == "__main__":
    main()
