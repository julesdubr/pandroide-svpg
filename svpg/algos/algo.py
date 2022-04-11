import torch as th
import torch.nn as nn

from salina import get_arguments, get_class, instantiate_class
from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace

from gym.spaces import Box, Discrete

from svpg.agents import ActionAgent, CriticAgent, CActionAgent, CCriticAgent
from svpg.agents.env import EnvAgentAutoReset
from svpg.common.logger import Logger


class Algo:
    def __init__(self, cfg):
        # --------------- Config infos --------------- #
        self.logger = Logger(cfg)
        self.env = instantiate_class(cfg.algorithm.env)

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

        # ---------  --------- #
        # Get the corresponding action/critic agents classes
        if isinstance(self.env.action_space, Discrete):
            actionAgent, criticAgent = ActionAgent, CriticAgent
        elif isinstance(self.env.action_space, Box):
            actionAgent, criticAgent = CActionAgent, CCriticAgent

        # -------------- Setup particles ------------- #
        self.env_agents = []
        self.action_agents = []
        self.critic_agents = []
        self.tcritic_agents = []
        self.acquisition_agents = []

        self.workspaces = []

        optimizer_args = get_arguments(cfg.algorithm.optimizer)
        self.optimizers = []

        for _ in range(self.n_particles):
            # Create envs
            env_agent = EnvAgentAutoReset(cfg)
            self.env_agents.append(env_agent)

            # Create agents
            action_agent = actionAgent(cfg, self.env)
            critic_agent = criticAgent(cfg, self.env)
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

    def execute_critic_agent(self):
        if not hasattr(self, "stop_variable"):
            for pid in range(self.n_particles):
                self.tcritic_agents[pid](self.workspaces[pid], n_steps=self.n_steps)
            return

        for pid in range(self.n_particles):
            self.tcritic_agents[pid](
                self.workspaces[pid], stop_variable=self.stop_variable
            )

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

    def compute_loss(self, workspaces, logger, epoch, alpha=10, verbose=True):
        # Needs to be defined by the child
        raise NotImplementedError

    def run(self, show_loss=False, show_grad=False):
        for epoch in range(self.max_epochs):
            # Run all particles
            self.execute_acquisition_agent(epoch)
            self.execute_critic_agent()

            # Compute loss
            critic_loss, entropy_loss, policy_loss, rewards = self.compute_loss(
                self.workspaces, self.logger, epoch, alpha=None, verbose=show_loss
            )

            loss = (
                -self.entropy_coef * entropy_loss
                + self.critic_coef * critic_loss
                + self.policy_coef * policy_loss
            )
            # Gradient descent
            for pid in range(self.n_particles):
                self.optimizers[pid].zero_grad()
            loss.backward()
            for pid in range(self.n_particles):
                self.optimizers[pid].step()

            # Log gradient norms
            if show_grad:
                self.compute_gradient_norm(epoch)

        return rewards
