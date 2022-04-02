from salina import instantiate_class, get_arguments, get_class
from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace
import torch, torch.nn as nn

from hydra.utils import instantiate

from logger import Logger

class Algo():
    def __init__(self, cfg):
        self.kernel = get_class(cfg.algorithm.kernel)
        self.logger = Logger(cfg)

        self.n_particles = cfg.algorithm.n_particles
        self.n_steps = cfg.algorithm.n_timesteps
        self.max_epochs = cfg.algorithm.max_epochs
        self.discount_factor = cfg.algorithm.discount_factor
        self.entropy_coef = cfg.algorithm.entropy_coef
        self.critic_coef = cfg.algorithm.critic_coef
        self.policy_coef = cfg.algorithm.policy_coef

        # Create agent
        self.action_agents = [instantiate_class(cfg.action_agent) for _ in range(self.n_particles)]
        self.critic_agents = [instantiate_class(cfg.critic_agent) for _ in range(self.n_particles)]
        self.env_agents = [instantiate_class(cfg.env_agent) for _ in range(self.n_particles)]

        self.tcritic_agents = [TemporalAgent(critic_agent) for critic_agent in self.critic_agents]
        self.acquisition_agents = [TemporalAgent(Agents(env_agent, action_agent)) 
                                  for env_agent, action_agent in zip(self.env_agents, self.action_agents)]
        for pid in range(self.n_particles):
            self.acquisition_agents[pid].seed(cfg.algorithm.env_seed)

        # Create workspace
        self.workspaces = [Workspace() for _ in range(self.n_particles)]

        # Setup optimizers
        optimizer_args = get_arguments(cfg.algorithm.optimizer)
        self.optimizers = []
        for pid in range(self.n_particles):
            params = nn.Sequential(self.action_agents[pid], self.critic_agents[pid]).parameters()
            self.optimizers.append(
                get_class(cfg.algorithm.optimizer)(
                    params, **optimizer_args
            ))

    def execute_acquisition_agent(self, epoch):
        for pid in range(self.n_particles):
            if epoch > 0:
                self.workspaces[pid].zero_grad()
                self.workspaces[pid].copy_n_last_steps(1)
                self.acquisition_agents[pid](
                    self.workspaces[pid], t=1, n_steps=self.n_steps - 1, stochastic=True
                )
            else:
                self.acquisition_agents[pid](self.workspaces[pid], t=0, n_steps=self.n_steps, stochastic=True)

    def execute_critic_agent(self):
        for pid in range(self.n_particles):
            self.tcritic_agents[pid](self.workspaces[pid], n_steps=self.n_steps)

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
        policy_loss.backward()

        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if j == i:
                    continue

                theta_i = self.action_agents[i].model.parameters()
                theta_j = self.action_agents[j].model.parameters()

                for (wi, wj) in zip(theta_i, theta_j):
                    wi.grad = wi.grad + wj.grad * kernel[j, i].detach()

    def compute_loss(self, epoch, alpha=10, verbose=True):
        # Need to defined in inherited classes
        raise NotImplementedError

    def run_svpg(self, alpha=10, show_losses=True, show_gradient=True):
        for epoch in range(self.max_epochs):
            # Run all particles
            self.execute_acquisition_agent(epoch)
            self.execute_critic_agent()

            # Compute loss
            policy_loss, critic_loss = self.compute_loss(epoch, alpha, show_losses)

            # Compute gradients
            thetas = self.get_policy_parameters()
            kernel = self.kernel()(thetas, thetas.detach())
            self.add_gradients(policy_loss, kernel)
            critic_loss.backward()
            
            # Gradient descent
            for pid in range(self.n_particles):
                self.optimizers[pid].step()

            for pid in range(self.n_particles):
                self.optimizers[pid].zero_grad()
    

    


    

    

        

            