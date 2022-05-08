import numpy as np
import torch
import torch.nn as nn

from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace

from gym.spaces import Discrete, Box

from svpg.agents import ActionAgent, CriticAgent, CActionAgent, CCriticAgent
from svpg.agents.env import NoAutoResetEnvAgent


from collections import defaultdict
from pathlib import Path
import os


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
        self.env_seed = env_seed
        self.rewards = defaultdict(lambda: [])
        self.eval_time_steps = defaultdict(lambda: [])
        self.eval_epoch = defaultdict(lambda: [])
        self.eval_interval = eval_interval
        self.clipped = clipped
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # --------------- Logger --------------- #
        self.logger = logger

        # --------------- Agents --------------- #
        self.train_env_agents = [
            env_agent(env_name, max_episode_steps, n_envs) for _ in range(n_particles)
        ]
        self.eval_env_agents = [
            NoAutoResetEnvAgent(env_name, max_episode_steps, 1)
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
                CCriticAgent(model(input_size, 1))
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
        for pid in range(self.n_particles):
            self.tcritic_agents[pid].to(self.device)
            self.train_acquisition_agents[pid].to(self.device)
            self.eval_acquisition_agents[pid].to(self.device)
            self.workspaces[pid].to(self.device)

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

    def save_best_agents(self, pid, directory):
        file_path = Path(str(directory) + "/agents/best_agent")
        torch.save(self.eval_acquisition_agents[pid].agent.agents[1], str(file_path))

    def save_all_agents(self, directory):
        critic_path = Path(str(directory) + "/agents/all_critic_agent")
        action_path = Path(str(directory) + "/agents/all_action_agent")

        if not os.path.exists(critic_path):
            os.makedirs(critic_path)
        if not os.path.exists(action_path):
            os.makedirs(action_path)

        for pid in range(self.n_particles):
            torch.save(
                self.critic_agents[pid], str(critic_path) + f"/critic_agent_{pid}.pt"
            )
            torch.save(
                self.eval_acquisition_agents[pid].agent.agents[1],
                str(action_path) + f"/action_agent_{pid}.pt",
            )

    def run(self, save_dir, max_grad_norm=0.5, show_loss=False, show_grad=False):
        self.to_gpu()
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
                +self.policy_coef * policy_loss / self.n_particles
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
                    eval_workspace = Workspace().to(self.device)
                    self.eval_acquisition_agents[pid](
                        eval_workspace, t=0, stop_variable="env/done", stochastic=False
                    )
                    creward, done = (
                        eval_workspace["env/cumulated_reward"],
                        eval_workspace["env/done"],
                    )
                    creward, done = creward.to(self.device), done.to(self.device)
                    tl = done.float().argmax(0)
                    creward = creward[tl, torch.arange(creward.size()[1])]
                    self.logger.add_log(f"reward_{pid}", creward.mean(), nb_steps[pid])
                    self.rewards[pid].append(creward.mean())
                    self.eval_time_steps[pid].append(nb_steps[pid])
                    self.eval_epoch[pid].append(epoch)

                last_epoch = epoch

        save_dir = Path(str(save_dir) + "/algo_base")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.save_all_agents(str(save_dir))

        reward_path = Path(str(save_dir) + "/rewards.npy")
        rewards_np = np.array(
            [[r.cpu() for r in agent_reward] for agent_reward in self.rewards.values()]
        )
        with open(reward_path, "wb") as f:
            np.save(f, rewards_np)
