from salina import get_arguments, get_class
from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace

import torch
import torch.nn as nn

from gym.spaces import Box, Discrete

from svpg.agents.discrete import ActionAgent, CriticAgent
from svpg.agents.continuous import ContinuousActionAgent, ContinuousCriticAgent
from svpg.agents.env import EnvAgent
from svpg.common.logger import Logger


class Algo:
    def __init__(self, cfg):
        self.logger = Logger(cfg)

        self.n_particles = cfg.algorithm.n_particles
        self.max_epochs = cfg.algorithm.max_epochs

        self.entropy_coef = cfg.algorithm.entropy_coef
        self.critic_coef = cfg.algorithm.critic_coef
        self.policy_coef = cfg.algorithm.policy_coef

        if not hasattr(self, "stop_variable"):
            try:
                self.n_steps = cfg.algorithm.n_timesteps
            except:
                raise NotImplemented

        # Setup particles
        self.env_agents = []
        self.action_agents = []
        self.critic_agents = []
        self.tcritic_agents = []
        self.acquisition_agents = []

        self.workspaces = []

        optimizer_args = get_arguments(cfg.algorithm.optimizer)
        self.optimizers = []

        for _ in range(self.n_particles):
            # Create agents
            env_agent = EnvAgent(cfg)
            if isinstance(env_agent.env.action_space, Discrete):
                action_agent = ActionAgent(cfg, env_agent.env)
                critic_agent = CriticAgent(cfg, env_agent.env)
            elif isinstance(env_agent.env.action_space, Box):
                action_agent = ContinuousActionAgent(cfg, env_agent.env)
                critic_agent = ContinuousCriticAgent(cfg, env_agent.env)
            del env_agent.env

            tacq_agent = TemporalAgent(Agents(env_agent, action_agent))
            tacq_agent.seed(cfg.algorithm.env_seed)

            self.env_agents.append(env_agent)
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

    def execute_acquisition_agent(self, epoch):
        for pid in range(self.n_particles):
            kwargs = {"workspace": self.workspaces[pid], "t": 0, "stochastic": True}

            if hasattr(self, "stop_variable"):
                if epoch > 0:
                    self.workspaces[pid].clear()
                kwargs["stop_variable"] = self.stop_variable

            else:
                kwargs["t"] = 1
                if epoch > 0:
                    self.workspaces[pid].copy_n_last_steps(1)
                    kwargs["n_steps"] = self.n_steps - 1
                else:
                    kwargs["n_steps"] = self.n_steps

            self.acquisition_agents[pid](**kwargs)

    def execute_critic_agent(self):
        for pid in range(self.n_particles):
            tcritic_agent = self.tcritic_agents[pid]
            if hasattr(self, "stop_variable"):
                tcritic_agent(self.workspaces[pid], stop_variable=self.stop_variable)
            else:
                tcritic_agent(self.workspaces[pid], n_steps=self.n_steps)

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
        # Need to defined in child classes
        raise NotImplementedError

    def run(self, show_loss=False, show_grad=False):
        for epoch in range(self.max_epochs):
            # Run all particles
            self.execute_acquisition_agent(epoch)
            self.execute_critic_agent()

            # Compute loss
            critic_loss, entropy_loss, policy_loss, rewards = self.compute_loss(
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

        return rewards
