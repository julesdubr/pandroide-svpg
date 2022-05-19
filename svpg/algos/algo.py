from pickletools import optimize
import numpy as np

import torch
from torch.nn.utils import clip_grad_norm_

from salina import get_arguments, get_class
from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace

from svpg.agents import ActionAgent, CriticAgent, ContinuousActionAgent
from svpg.agents.env import AutoResetEnvAgent, NoAutoResetEnvAgent, get_env_infos
from svpg.utils.utils import save_algo
from svpg.logger import Logger

from collections import defaultdict


class Algo:
    def __init__(self, cfg, solo=False):
        self.cfg = cfg

        self.n_particles = 1 if solo else cfg.algorithm.n_particles
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
        self.eval_timesteps = []

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        self.logger = Logger(cfg)

        self.action_agents = []
        self.critic_agents = []

        self.train_agents = []
        self.eval_agents = []

        self.tcritic_agents = []

        self.train_workspaces = []

        self.action_optimizers = []
        self.critic_optimizers = []

        for _ in range(self.n_particles):
            train_env_agent = AutoResetEnvAgent(cfg, self.n_envs)
            eval_env_agent = NoAutoResetEnvAgent(cfg, self.n_evals)

            (_, obs_size), (is_continuous, n_actions) = get_env_infos(train_env_agent)
            hidden_size = cfg.algorithm.architecture.hidden_size

            actionAgentClass = ContinuousActionAgent if is_continuous else ActionAgent
            action_agent = actionAgentClass(obs_size, hidden_size, n_actions)

            self.action_agents.append(action_agent)

            critic_agent = CriticAgent(obs_size, cfg.algorithm.architecture.hidden_size)
            self.critic_agents.append(critic_agent)

            # Get an agent that is executed on a complete workspace
            train_agent = TemporalAgent(Agents(train_env_agent, action_agent))
            train_agent.seed(cfg.algorithm.seed)
            self.train_agents.append(train_agent)

            self.eval_agents.append(TemporalAgent(Agents(eval_env_agent, action_agent)))

            self.tcritic_agents.append(TemporalAgent(critic_agent))

            self.train_workspaces.append(Workspace())

            # Optimizers
            optimizer = get_class(cfg.optimizer)
            optimizer_args = get_arguments(cfg.optimizer)

            self.action_optimizers.append(
                optimizer(action_agent.parameters(), **optimizer_args)
            )
            self.critic_optimizers.append(
                optimizer(critic_agent.parameters(), **optimizer_args)
            )

    def _execute_train_agents(self, epoch):
        for pid in range(self.n_particles):
            kwargs = {"t": 0, "stochastic": True, "n_steps": self.n_steps}
            if epoch > 0:
                self.train_workspaces[pid].zero_grad()
                self.train_workspaces[pid].copy_n_last_steps(1)
                kwargs["t"] = 1
                kwargs["n_steps"] = self.n_steps - 1

            self.train_agents[pid](self.train_workspaces[pid], **kwargs)

    def _execute_tcritic_agents(self):
        if self.critic_coef == 0:
            return

        for pid in range(self.n_particles):
            self.tcritic_agents[pid](self.train_workspaces[pid], n_steps=self.n_steps)

    def _compute_gradient_norm(self, epoch):
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

    def _compute_loss(self, epoch, verbose=True):
        # Needs to be defined by the child
        raise NotImplementedError

    def run(self, save_dir, max_gradn=0.6, show_loss=False, show_grad=False):
        action_loss = 0
        tmp_epoch = 0
        steps = 0

        for epoch in range(self.max_epochs):
            # Run all particles
            self._execute_train_agents(epoch)
            self._execute_tcritic_agents()

            steps += self.n_steps * self.n_envs

            # Compute loss
            policy_loss, critic_loss, entropy_loss = self._compute_loss(
                epoch, verbose=show_loss
            )

            critic_loss = self.critic_coef * critic_loss / self.n_particles
            critic_loss.backward()

            if self.clipped:
                for critic_agent in self.critic_agents:
                    clip_grad_norm_(critic_agent.parameters(), max_gradn)

            # Critic gradient descent
            for critic_optimizer in self.critic_optimizers:
                critic_optimizer.step()
            for critic_optimizer in self.critic_optimizers:
                critic_optimizer.zero_grad()

            action_loss = action_loss + (
                self.policy_coef * policy_loss / self.n_particles
                + self.entropy_coef * entropy_loss / self.n_particles
            )

            # Evaluation
            if epoch - tmp_epoch > self.eval_interval:
                tmp_epoch = epoch
                self.eval_timesteps.append(steps)

                action_loss.backward()
                action_loss = 0

                if self.clipped:
                    for action_agent in self.action_agents:
                        clip_grad_norm_(action_agent.parameters(), max_gradn)

                # Log gradient norms
                if show_grad:
                    self._compute_gradient_norm(epoch)

                # Actor gradient descent
                for action_optimizer in self.action_optimizers:
                    action_optimizer.step()
                for action_optimizer in self.action_optimizers:
                    action_optimizer.zero_grad()

                for pid in range(self.n_particles):
                    eval_workspace = Workspace()
                    self.eval_agents[pid](
                        eval_workspace, t=0, stop_variable="env/done", stochastic=False
                    )
                    rewards = eval_workspace["env/cumulated_reward"][-1]
                    mean = rewards.mean()
                    # print("eval rewards:", rewards)
                    # print("eval mean   :", mean)
                    self.logger.add_log(f"reward_{pid}", mean, steps)
                    self.rewards[pid].append(mean)

        if self.cfg.save_run:
            save_algo(self, save_dir)
