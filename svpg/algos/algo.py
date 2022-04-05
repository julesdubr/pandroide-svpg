from salina import instantiate_class, get_arguments, get_class
from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace

import torch
import torch.nn as nn

import numpy as np

from svpg.common.logger import Logger


class Algo:
    def __init__(self, cfg):
        self.kernel = get_class(cfg.algorithm.kernel)
        self.logger = Logger(cfg)
        self.stop_variable = None

        self.n_particles = cfg.algorithm.n_particles
        try:
            self.n_steps = cfg.algorithm.n_timesteps
        except:
            self.n_steps = None
        self.max_epochs = cfg.algorithm.max_epochs
        self.discount_factor = cfg.algorithm.discount_factor
        self.entropy_coef = cfg.algorithm.entropy_coef
        self.critic_coef = cfg.algorithm.critic_coef
        self.policy_coef = cfg.algorithm.policy_coef

        # Create agents
        self.action_agents, self.critic_agents, self.env_agents = list(), list(), list()
        for _ in range(self.n_particles):
            self.action_agents.append(instantiate_class(cfg.action_agent))
            self.critic_agents.append(instantiate_class(cfg.critic_agent))
            self.env_agents.append(instantiate_class(cfg.env_agent))

        self.tcritic_agents = [
            TemporalAgent(critic_agent) for critic_agent in self.critic_agents
        ]
        self.acquisition_agents = [
            TemporalAgent(Agents(env_agent, action_agent))
            for env_agent, action_agent in zip(self.env_agents, self.action_agents)
        ]
        for pid in range(self.n_particles):
            self.acquisition_agents[pid].seed(cfg.algorithm.env_seed)

        # Create workspaces
        self.workspaces = [Workspace() for _ in range(self.n_particles)]

        # Setup optimizers
        optimizer_args = get_arguments(cfg.algorithm.optimizer)
        self.optimizers = []
        for pid in range(self.n_particles):
            params = nn.Sequential(
                self.action_agents[pid], self.critic_agents[pid]
            ).parameters()
            self.optimizers.append(
                get_class(cfg.algorithm.optimizer)(params, **optimizer_args)
            )

        self.rewards = np.zeros(self.n_particles)

    def execute_acquisition_agent(self, epoch):
        for pid in range(self.n_particles):
            if epoch > 0:
                self.workspaces[pid].zero_grad()
                if self.stop_variable is None:
                    self.workspaces[pid].copy_n_last_steps(1)
                    self.acquisition_agents[pid](
                        self.workspaces[pid],
                        t=1,
                        n_steps=self.n_steps - 1,
                        stochastic=True,
                    )
                else:
                    self.workspaces[pid].clear()
                    self.acquisition_agents[pid](
                        self.workspaces[pid],
                        t=0,
                        stop_variable=self.stop_variable,
                        stochastic=True,
                    )
            else:
                if self.stop_variable is None:
                    self.acquisition_agents[pid](
                        self.workspaces[pid], t=0, n_steps=self.n_steps, stochastic=True
                    )
                else:
                    self.acquisition_agents[pid](
                        self.workspaces[pid],
                        t=0,
                        stop_variable=self.stop_variable,
                        stochastic=True,
                    )

    def execute_critic_agent(self):
        for pid in range(self.n_particles):
            if self.stop_variable is None:
                self.tcritic_agents[pid](self.workspaces[pid], n_steps=self.n_steps)
            else:
                self.tcritic_agents[pid](
                    self.workspaces[pid], stop_variable=self.stop_variable
                )

    def get_policy_parameters(self):
        policy_params = []
        for pid in range(self.n_particles):
            l = list(self.action_agents[pid].model.parameters())
            l_flatten = [torch.flatten(p) for p in l]
            l_flatten = tuple(l_flatten)
            l_concat = torch.cat(l_flatten)

            policy_params.append(l_concat)

        return torch.stack(policy_params)

    def add_gradients(self, policy_loss, kernel):
        policy_loss.backward(retain_graph=True)

        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if j == i:
                    continue

                theta_i = self.action_agents[i].model.parameters()
                theta_j = self.action_agents[j].model.parameters()

                for (wi, wj) in zip(theta_i, theta_j):
                    wi.grad = wi.grad + wj.grad * kernel[j, i].detach()

    def compute_gradient_norm(self, epoch):
        policy_gradnorm, critic_gradnorm = 0, 0

        for pid in range(self.n_particles):
            # prob_params = particle["prob_agent"].model.parameters()
            # critic_params = particle["critic_agent"].critic_model.parameters()
            policy_params = self.action_agents[pid].model.parameters()
            critic_params = self.critic_agents[pid].model.parameters()

            for w_policy, w_critic in zip(policy_params, critic_params):
                if w_policy.grad != None:
                    policy_gradnorm += w_policy.grad.detach().data.norm(2) ** 2

                if w_critic.grad != None:
                    critic_gradnorm += w_critic.grad.detach().data.norm(2) ** 2

        policy_gradnorm, critic_gradnorm = (
            torch.sqrt(policy_gradnorm),
            torch.sqrt(critic_gradnorm),
        )

        self.logger.add_log("Policy Gradient norm", policy_gradnorm, epoch)
        self.logger.add_log("Critic Gradient norm", critic_gradnorm, epoch)

    def compute_loss(self, epoch, alpha=10, verbose=True):
        # Need to defined in inherited classes
        raise NotImplementedError

    def run_svpg(self, alpha=10, show_loss=False, show_grad=False):
        for epoch in range(self.max_epochs):
            # Run all particles
            self.execute_acquisition_agent(epoch)
            self.execute_critic_agent()

            # Compute loss
            critic_loss, entropy_loss, policy_loss = self.compute_loss(
                epoch, alpha, show_loss
            )

            # print(critic_loss)

            # Compute gradients
            thetas = self.get_policy_parameters()
            kernel = self.kernel()(thetas, thetas.detach())
            self.add_gradients(policy_loss, kernel)

            loss = (
                -self.entropy_coef * entropy_loss
                + self.critic_coef * critic_loss
                + kernel.sum() / self.n_particles
            )
            loss.backward()
            # Log gradient norms

            if show_grad:
                self.compute_gradient_norm(epoch)

            # Gradient descent
            for pid in range(self.n_particles):
                self.optimizers[pid].step()
                self.optimizers[pid].zero_grad()

    def run(self, show_loss=False, show_grad=False):
        for epoch in range(self.max_epochs):
            # Run all particles
            self.execute_acquisition_agent(epoch)
            self.execute_critic_agent()

            # Compute loss
            critic_loss, entropy_loss, policy_loss = self.compute_loss(
                epoch, alpha=None, verbose=show_loss
            )

            loss = (
                -self.entropy_coef * entropy_loss
                + self.critic_coef * critic_loss
                + self.policy_coef * policy_loss
            )
            loss.backward()

            # Log gradient norms
            if show_grad:
                self.compute_gradient_norm(epoch)

            # Gradient descent
            for pid in range(self.n_particles):
                self.optimizers[pid].step()
                self.optimizers[pid].zero_grad()
