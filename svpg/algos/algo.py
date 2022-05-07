import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from salina import get_arguments, get_class
from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace

from svpg.agents import ActionAgent, CriticAgent, ContinuousActionAgent
from svpg.agents.env import AutoResetEnvAgent, NoAutoResetEnvAgent
from svpg.utils.utils import save_algo_data
from svpg.logger import Logger

from rllab.spaces import Discrete, Box

from collections import defaultdict


class Algo:
    def __init__(self, cfg):
        self.cfg = cfg

        self.n_particles = cfg.algorithm.n_particles
        self.clipped = cfg.algorithm.clipped
        self.n_steps = cfg.algorithm.n_steps
        self.n_evals = cfg.algorithm.n_evals
        self.n_envs = cfg.algorithm.n_envs
        self.max_epochs = cfg.algorithm.max_epochs
        self.eval_interval = cfg.algorithm.eval_interval

        self.policy_coef = cfg.algorithm.policy_coef
        self.entropy_coef = cfg.algorithm.entropy_coef
        self.critic_coef = cfg.algorithm.critic_coef

        self.rewards = defaultdict(lambda: [])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.logger = Logger(cfg)

        self.action_agents = []
        self.critic_agents = []

        self.train_agents = []
        self.eval_agents = []

        self.tcritic_agents = []

        self.train_workspaces = []
        self.optimizers = []

        for _ in range(self.n_particles):
            train_env_agent = AutoResetEnvAgent(cfg, n_envs=self.n_envs)
            eval_env_agent = NoAutoResetEnvAgent(cfg, n_envs=self.n_evals)

            observation_size, n_actions = train_env_agent.get_obs_and_actions_sizes()

            if train_env_agent.is_continuous_action() or isinstance(
                train_env_agent.action_space, Box
            ):
                action_agent = ContinuousActionAgent(
                    observation_size, cfg.algorithm.architecture.hidden_size, n_actions
                )
            else:
                action_agent = ActionAgent(
                    observation_size, cfg.algorithm.architecture.hidden_size, n_actions
                )

            self.action_agents.append(action_agent)

            critic_agent = CriticAgent(
                observation_size, cfg.algorithm.architecture.hidden_size
            )
            self.critic_agents.append(critic_agent)

            # Get an agent that is executed on a complete workspace
            train_agent = TemporalAgent(Agents(train_env_agent, action_agent))
            train_agent.seed(cfg.algorithm.seed)
            self.train_agents.append(train_agent)

            self.eval_agents.append(TemporalAgent(Agents(eval_env_agent, action_agent)))

            self.tcritic_agents.append(TemporalAgent(critic_agent))

            self.train_workspaces.append(Workspace())

            optimizer_args = get_arguments(cfg.optimizer)
            parameters = nn.Sequential(action_agent, critic_agent).parameters()
            self.optimizers.append(
                get_class(cfg.optimizer)(parameters, **optimizer_args)
            )

    def execute_train_agents(self, epoch):
        for pid in range(self.n_particles):
            kwargs = {"t": 0, "stochastic": True, "n_steps": self.n_steps}
            if epoch > 0:
                self.train_workspaces[pid].zero_grad()
                self.train_workspaces[pid].copy_n_last_steps(1)
                kwargs["t"] = 1
                kwargs["n_steps"] = self.n_steps - 1

            self.train_agents[pid](self.train_workspaces[pid], **kwargs)

    def execute_tcritic_agents(self):
        if self.critic_coef == 0:
            return

        for pid in range(self.n_particles):
            self.tcritic_agents[pid](self.train_workspaces[pid], n_steps=self.n_steps)

    def to_device(self):
        for pid in range(self.n_particles):
            self.tcritic_agents[pid].to(self.device)
            self.train_agents[pid].to(self.device)
            self.eval_agents[pid].to(self.device)
            self.train_workspaces[pid].to(self.device)

    def compute_gradient_norm(self, epoch):
        policy_gradnorm, critic_gradnorm = 0, 0

        for pid in range(self.n_particles):
            policy_params = self.action_agents[pid].parameters()
            critic_params = self.critic_agents[pid].parameters()

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

    def run(self, save_dir, max_grad_norm=0.5, show_loss=False, show_grad=False):
        self.to_device()

        nb_steps = 0
        tmp_steps = 0

        for epoch in range(self.max_epochs):
            # Run all particles
            self.execute_train_agents(epoch)
            self.execute_tcritic_agents()

            nb_steps += self.n_steps * self.n_envs

            # Compute loss
            policy_loss, critic_loss, entropy_loss = self.compute_loss(
                epoch, verbose=show_loss
            )

            loss = (
                +self.policy_coef * policy_loss / self.n_particles
                + self.critic_coef * critic_loss / self.n_particles
                + self.entropy_coef * entropy_loss / self.n_particles
            )

            if self.clipped:
                for pid in range(self.n_particles):
                    clip_grad_norm_(self.action_agents[pid].parameters(), max_grad_norm)
                    clip_grad_norm_(self.critic_agents[pid].parameters(), max_grad_norm)

            # Log gradient norms
            if show_grad:
                self.compute_gradient_norm(epoch)

            # Gradient descent
            for pid in range(self.n_particles):
                self.optimizers[pid].zero_grad()
            loss.backward()
            for pid in range(self.n_particles):
                self.optimizers[pid].step()

            # Evaluation
            if nb_steps - tmp_steps > self.eval_interval:
                tmp_steps = nb_steps

                for pid in range(self.n_particles):
                    eval_workspace = Workspace().to(self.device)
                    self.eval_agents[pid](
                        eval_workspace, t=0, stop_variable="env/done", stochastic=False
                    )
                    rewards = eval_workspace["env/cumulated_reward"][-1]
                    mean = rewards.mean()
                    self.logger.add_log(f"reward_{pid}", mean, nb_steps)
                    self.rewards[pid].append(mean)

        save_algo_data(self, save_dir)
