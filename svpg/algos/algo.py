from functools import partial

import torch as th
import torch.nn as nn

from salina import get_arguments, get_class
from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace

from gym.spaces import Discrete

from svpg.agents import ActionAgent, CriticAgent, CActionAgent, CCriticAgent
from svpg.agents.env import make_env


class Algo:
    def __init__(self, 
                 n_particles, 
                 max_epochs, discount_factor,
                 env_name, max_episode_steps, n_envs, env_seed,
                 logger,
                 env_agent,
                 env, 
                 model, 
                 optimizer):
        # --------------- Hyper parameters --------------- #
        self.n_particles = n_particles
        self.max_epochs = max_epochs
        self.discount_factor = discount_factor
        self.n_env = n_envs

        # --------------- Logger --------------- #
        self.logger = logger

        # --------------- Agents --------------- #
        self.env_agents = [env_agent(env_name, max_episode_steps, n_envs) for _ in range(n_particles)]

        if isinstance(env.action_space, Discrete):
            input_size, output_size = env.observation_space.shape[0], env.action_space.n
            self.action_agents = [ActionAgent(model(input_size, output_size)) for _ in range(n_particles)]
            self.critic_agents = [CriticAgent(model(input_size, 1)) for _ in range(n_particles)]
        else:
            input_size, output_size = env.observation_space.shape[0], env.action_space.shape[0]
            self.action_agents = [CActionAgent(output_size, model(input_size, output_size)) for _ in range(n_particles)]
            self.critic_agents = [CriticAgent(model(input_size, 1, activation=nn.SiLU)) for _ in range(n_particles)]

        self.tcritic_agents = [TemporalAgent(critic_agent) for critic_agent in self.critic_agents]
        self.acquisition_agents = [TemporalAgent(Agents(env_agent, action_agent)) for env_agent, action_agent in zip(self.env_agents, self.action_agents)]
        
        for acquisition_agent in self.acquisition_agents:
            acquisition_agent.seed(env_seed)

        # ---------------- Workspaces ------------ #
        self.workspaces = [Workspace() for _ in range(n_particles)]

        # ---------------- Optimizers ------------ #
        self.optimizers = [optimizer(nn.Sequential(action_agent, critic_agent).parameters()) 
                           for action_agent, critic_agent in zip(self.action_agents, self.critic_agents)]

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
            policy_params = action_agent.parameters()
            critic_params = critic_agent.parameters()

            for w in policy_params:
                if w.grad != None:
                    policy_gradnorm += w.grad.detach().data.norm(2) ** 2

            for w in critic_params:
                if w.grad != None:
                    critic_gradnorm += w.grad.detach().data.norm(2) ** 2

        policy_gradnorm, critic_gradnorm = th.sqrt(policy_gradnorm), th.sqrt(critic_gradnorm)

        self.logger.add_log("Policy Gradient norm", policy_gradnorm, epoch)
        self.logger.add_log("Critic Gradient norm", critic_gradnorm, epoch)

    def compute_loss(self, epoch, verbose=True):
        # Needs to be defined by the child
        raise NotImplementedError

    def run(self, show_loss=True, show_grad=True):
        for epoch in range(self.max_epochs):
            # Run all particles
            self.execute_acquisition_agent(epoch)
            self.execute_critic_agent()

            # Compute loss
            policy_loss, critic_loss, entropy_loss, rewards = self.compute_loss(
                epoch, verbose=show_loss
            )

            total_loss = (
                + self.policy_coef * policy_loss
                + self.critic_coef * critic_loss
                + self.entropy_coef * entropy_loss
            )

            for pid in range(self.n_particles):
                self.optimizers[pid].zero_grad()
            
            total_loss.backward()

            # Log gradient norms
            if show_grad:
                self.compute_gradient_norm(epoch)
            
            # Gradient descent
            for pid in range(self.n_particles):
                self.optimizers[pid].step()

        return rewards
