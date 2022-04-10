from salina import instantiate_class, get_arguments, get_class
from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace

import torch as th
import torch.nn as nn

from torch.nn.utils import parameters_to_vector

import numpy as np

from svpg.common.logger import Logger
from svpg.agents.discrete import ActionAgent, CriticAgent
from svpg.agents.env import EnvAgentAutoReset
from svpg.kernel import RBF

from itertools import permutations


class Algo:
    def __init__(self, cfg):
        self.kernel = RBF
        self.logger = Logger(cfg)

        self.n_particles = cfg.algorithm.n_particles
        self.max_epochs = cfg.algorithm.max_epochs

        self.discount_factor = cfg.algorithm.discount_factor
        self.entropy_coef = cfg.algorithm.entropy_coef
        self.critic_coef = cfg.algorithm.critic_coef
        self.policy_coef = cfg.algorithm.policy_coef

        if not hasattr(self, "stop_variable"):
            try:
                self.n_steps = cfg.algorithm.n_timesteps
            except:
                raise ValueError

        # --------- Setup environment agents --------- #
        self.env_agents = [EnvAgentAutoReset(cfg) for _ in range(self.n_particles)]

        # -------------- Setup particles ------------- #
        self.action_agents = []
        self.critic_agents = []
        self.tcritic_agents = []
        self.acquisition_agents = []

        self.workspaces = []

        optimizer_args = get_arguments(cfg.algorithm.optimizer)
        self.optimizers = []

        for env_agent in self.env_agents:
            # Create agents
            action_agent = ActionAgent(cfg, env_agent.env)
            critic_agent = CriticAgent(cfg, env_agent.env)
            del env_agent.env

            tacq_agent = TemporalAgent(Agents(env_agent, action_agent))
            tacq_agent.seed(cfg.algorithm.env_seed)

            self.action_agents.append(action_agent)
            self.critic_agents.append(critic_agent)
            self.tcritic_agents.append(TemporalAgent(critic_agent))
            self.acquisition_agents.append(tacq_agent)

            # Create workspaces
            self.workspaces.append(Workspace())

            # Create optimizers
            params = nn.Sequential(action_agent, critic_agent).parameters()
            self.optimizers.append(
                get_class(cfg.algorithm.optimizer)(params, **optimizer_args)
            )

        self.rewards = np.zeros(self.n_particles)

    def execute_acquisition_agent(self, epoch):
        if not hasattr(self, "stop_variable"):
            for pid in range(self.n_particles):
                kwargs = {"t": 0, "stochastic": True, "n_steps": self.n_steps}
                if epoch > 0:
                    self.workspaces[pid].zero_grad()
                    self.workspaces[pid].copy_n_last_steps(1)
                    kwargs["t"] = 1
                    kwargs["n_steps"] = self.n_steps - 1

                self.acquisition_agents[pid](self.workspaces[pid], **kwargs)
            return

        for pid in range(self.n_particles):
            kwargs = {"t": 0, "stochastic": True, "stop_variable": self.stop_variable}
            if epoch > 0:
                self.workspaces[pid].zero_grad()
                self.workspaces[pid].clear()

            self.acquisition_agents[pid](self.workspaces[pid], **kwargs)

    # def execute_acquisition_agent(self, epoch):
    #     for pid in range(self.n_particles):
    #         if epoch > 0:
    #             self.workspaces[pid].zero_grad()
    #             if not hasattr(self, "stop_variable"):
    #                 self.workspaces[pid].copy_n_last_steps(1)
    #                 self.acquisition_agents[pid](
    #                     self.workspaces[pid],
    #                     t=1,
    #                     n_steps=self.n_steps - 1,
    #                     stochastic=True,
    #                 )
    #             else:
    #                 self.workspaces[pid].clear()
    #                 self.acquisition_agents[pid](
    #                     self.workspaces[pid],
    #                     t=0,
    #                     stop_variable=self.stop_variable,
    #                     stochastic=True,
    #                 )
    #         else:
    #             if not hasattr(self, "stop_variable"):
    #                 self.acquisition_agents[pid](
    #                     self.workspaces[pid], t=0, n_steps=self.n_steps, stochastic=True
    #                 )
    #             else:
    #                 self.acquisition_agents[pid](
    #                     self.workspaces[pid],
    #                     t=0,
    #                     stop_variable=self.stop_variable,
    #                     stochastic=True,
    #                 )

    def execute_critic_agent(self):
        if not hasattr(self, "stop_variable"):
            for pid in range(self.n_particles):
                self.tcritic_agents[pid](self.workspaces[pid], n_steps=self.n_steps)
            return

        for pid in range(self.n_particles):
            self.tcritic_agents[pid](
                self.workspaces[pid], stop_variable=self.stop_variable
            )

    def get_policy_parameters(self):
        policy_params = [
            parameters_to_vector(action_agent.model.parameters())
            for action_agent in self.action_agents
        ]
        return th.stack(policy_params)

    def add_gradients(self, policy_loss, kernel):
        policy_loss.backward(retain_graph=True)

        # Get all the couples of particules (i,j) st. i /= j
        for i, j in list(permutations(range(self.n_particles), r=2)):

            theta_i = self.action_agents[i].model.parameters()
            theta_j = self.action_agents[j].model.parameters()

            for (wi, wj) in zip(theta_i, theta_j):
                wi.grad = wi.grad + wj.grad * kernel[j, i].detach()

    def compute_gradient_norm(self, epoch):
        policy_gradnorm, critic_gradnorm = 0, 0

        for action_agent, critic_agent in zip(self.action_agents, self.critic_agents):
            policy_params = action_agent.model.parameters()
            critic_params = critic_agent.model.parameters()

            for w in policy_params + critic_params:
                if w.grad != None:
                    policy_gradnorm += w.grad.detach().data.norm(2) ** 2

        policy_gradnorm, critic_gradnorm = (
            th.sqrt(th.stack([policy_gradnorm, critic_gradnorm])),
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
