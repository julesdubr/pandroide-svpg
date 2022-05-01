import torch
import torch.nn as nn

from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace

from gym.spaces import Discrete

from svpg.agents import ActionAgent, CriticAgent, CActionAgent, CCriticAgent
from svpg.agents.env import EnvAgentNoAutoReset

import numpy as np

from collections import defaultdict


class Algo:
    def __init__(
        self,
        n_particles,
        max_epochs,
        discount_factor,
        env_name,
        max_episode_steps,
        n_envs,
        env_seed,
        eval_interval,
        clipped,
        logger,
        env_agent,
        env,
        model,
        optimizer,
    ):
        # --------------- Hyper parameters --------------- #
        self.n_particles = n_particles
        self.max_epochs = max_epochs
        self.discount_factor = discount_factor
        self.n_env = n_envs
        self.rewards = defaultdict(lambda: [])
        self.eval_time_steps = defaultdict(lambda: [])
        self.eval_interval = eval_interval
        self.clipped = clipped

        # --------------- Logger --------------- #
        self.logger = logger

        # --------------- Agents --------------- #
        self.train_env_agents = [
            env_agent(env_name, max_episode_steps, n_envs) for _ in range(n_particles)
        ]
        self.eval_env_agents = [
            EnvAgentNoAutoReset(env_name, max_episode_steps, n_envs)
            for _ in range(n_particles)
        ]

        if isinstance(env.action_space, Discrete):
            input_size, output_size = env.observation_space.shape[0], env.action_space.n
            self.action_agents = [
                ActionAgent(model(input_size, output_size)) for _ in range(n_particles)
            ]
            self.critic_agents = [
                CriticAgent(model(input_size, 1)) for _ in range(n_particles)
            ]
        else:
            input_size, output_size = (
                env.observation_space.shape[0],
                env.action_space.shape[0],
            )
            self.action_agents = [
                CActionAgent(output_size, model(input_size, output_size))
                for _ in range(n_particles)
            ]
            self.critic_agents = [
                CCriticAgent(model(input_size, 1, activation=nn.SiLU))
                for _ in range(n_particles)
            ]

        self.tcritic_agents = [
            TemporalAgent(critic_agent) for critic_agent in self.critic_agents
        ]
        self.train_acquisition_agents = [
            TemporalAgent(Agents(train_env_agent, action_agent))
            for train_env_agent, action_agent in zip(
                self.train_env_agents, self.action_agents
            )
        ]
        self.eval_acquisition_agents = [
            TemporalAgent(Agents(eval_env_agent, action_agent))
            for eval_env_agent, action_agent in zip(
                self.eval_env_agents, self.action_agents
            )
        ]

        for train_acquisition_agent in self.train_acquisition_agents:
            train_acquisition_agent.seed(env_seed)

        # ---------------- Workspaces ------------ #
        self.workspaces = [Workspace() for _ in range(n_particles)]

        # ---------------- Optimizers ------------ #
        self.optimizers = [
            optimizer(nn.Sequential(action_agent, critic_agent).parameters())
            for action_agent, critic_agent in zip(
                self.action_agents, self.critic_agents
            )
        ]

    def execute_acquisition_agent(self, epoch):
        if not hasattr(self, "stop_variable"):
            for pid in range(self.n_particles):
                kwargs = {"t": 0, "stochastic": True, "n_steps": self.n_steps}
                if epoch > 0:
                    self.workspaces[pid].zero_grad()
                    self.workspaces[pid].copy_n_last_steps(1)
                    kwargs["t"] = 1
                    kwargs["n_steps"] = self.n_steps - 1

                self.train_acquisition_agents[pid](self.workspaces[pid], **kwargs)
            return

        for pid in range(self.n_particles):
            kwargs = {"t": 0, "stochastic": True, "stop_variable": self.stop_variable}
            if epoch > 0:
                self.workspaces[pid].clear()

            self.train_acquisition_agents[pid](self.workspaces[pid], **kwargs)

    def execute_critic_agent(self):
        if not hasattr(self, "stop_variable"):
            for pid in range(self.n_particles):
                self.tcritic_agents[pid](self.workspaces[pid], n_steps=self.n_steps)
            return

        for pid in range(self.n_particles):
            self.tcritic_agents[pid](
                self.workspaces[pid], stop_variable=self.stop_variable
            )

    def to_gpu(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            for pid in range(self.n_particles):
                self.tcritic_agents[pid].to(device)
                self.train_acquisition_agents[pid].to(device)
                self.eval_acquisition_agents[pid].to(device)

    def compute_gradient_norm(self, epoch):
        policy_gradnorm, critic_gradnorm = 0, 0

        for action_agent, critic_agent in zip(self.action_agents, self.critic_agents):
            policy_params = action_agent.parameters()
            critic_params = critic_agent.parameters()

            for w in policy_params:
                if w.grad != None:
                    policy_gradnorm += w.grad.detach().data.norm(2).item() ** 2

            for w in critic_params:
                if w.grad != None:
                    critic_gradnorm += w.grad.detach().data.norm(2).item() ** 2

        policy_gradnorm, critic_gradnorm = np.sqrt(policy_gradnorm), np.sqrt(
            critic_gradnorm
        )

        self.logger.add_log("Policy Gradient norm", policy_gradnorm, epoch)
        self.logger.add_log("Critic Gradient norm", critic_gradnorm, epoch)

    def compute_loss(self, epoch, verbose=True):
        # Needs to be defined by the child
        raise NotImplementedError

    def run(self, max_grad_norm=0.5, show_loss=False, show_grad=False):
        nb_steps = np.zeros(self.n_particles)
        last_epoch = 0
        for epoch in range(self.max_epochs):
            # Run all particles
            self.execute_acquisition_agent(epoch)
            if self.critic_coef != 0:
                self.execute_critic_agent()

            # Compute loss
            policy_loss, critic_loss, entropy_loss, n_steps = self.compute_loss(
                epoch, verbose=show_loss
            )

            total_loss = (
                + self.policy_coef * policy_loss / self.n_particles
                + self.critic_coef * critic_loss / self.n_particles
                + self.entropy_coef * entropy_loss / self.n_particles
            )

            for pid in range(self.n_particles):
                self.optimizers[pid].zero_grad()

            total_loss.backward()

            if self.clipped:
                for pid in range(self.n_particles):
                    torch.nn.utils.clip_grad_norm_(
                        self.action_agents[pid].parameters(), max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.critic_agents[pid].parameters(), max_grad_norm
                    )

            # Log gradient norms
            if show_grad:
                self.compute_gradient_norm(epoch)

            # Gradient descent
            for pid in range(self.n_particles):
                self.optimizers[pid].step()

            # Gradient descent
            for pid in range(self.n_particles):
                self.optimizers[pid].zero_grad()

            # Evaluation
            nb_steps += n_steps
            if epoch - last_epoch == self.eval_interval - 1:
                for pid in range(self.n_particles):
                    eval_workspace = Workspace()
                    self.eval_acquisition_agents[pid](
                        eval_workspace, t=0, stop_variable="env/done", stochastic=False
                    )
                    creward, done = (
                        eval_workspace["env/cumulated_reward"],
                        eval_workspace["env/done"],
                    )
                    tl = done.float().argmax(0)
                    creward = creward[tl, torch.arange(creward.size()[1])]
                    self.logger.add_log(f"reward_{pid}", creward.mean(), nb_steps[pid])
                    self.rewards[pid].append(creward.mean())
                    self.eval_time_steps[pid].append(nb_steps[pid])

                last_epoch = epoch
