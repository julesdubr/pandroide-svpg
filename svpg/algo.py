from salina import instantiate_class, get_arguments, get_class
from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace
import torch, torch.nn as nn

from hydra.utils import instantiate

class Algo():
    def __init__(self, **kwargs):
        self.n_particles = kwargs["n_particles"]
        self.n_steps = kwargs["n_timesteps"]
        self.kernel = get_class(kwargs["kernel"])
        self.max_epochs = kwargs["max_epochs"]

        # Create agent
        self.action_agents, self.critic_agents, self.env_agents, self.acquisition_agents, self.tcritic_agents = [], [], [], [], []
        for pid in range(self.n_particles):
        #     print(kwargs["action_agent"])
        #     instantiate(kwargs["action_agent"], _recursive_=False)

        # self.action_agents = [instantiate_class(kwargs["action_agent"]) for _ in range(self.n_particles)]
        # self.critic_agents = [instantiate_class(kwargs["critic_agent"]) for _ in range(self.n_particles)]
        # self.tcritic_agents = [TemporalAgent(critic_agent) for critic_agent in self.critic_agents]
        # self.env_agents = [instantiate_class(kwargs["env"]) for _ in range(self.n_particles)]
            action_agent = instantiate_class(kwargs["action_agent"])
            critic_agent = instantiate_class(kwargs["critic_agent"])
            env_agent = instantiate_class(kwargs["env_agent"])

            self.action_agents.append(action_agent)
            self.critic_agents.append(critic_agent)
            self.env_agents.append(env_agent)

            self.tcritic_agents.append(TemporalAgent(critic_agent))
            self.acquisition_agents.append(TemporalAgent(Agents(env_agent, action_agent)))
            self.acquisition_agents[pid].seed(kwargs["env_seed"])

        # self.acquisition_agents = []
        # for pid in range(self.n_particles):
        #     self.acquisition_agents.append(TemporalAgent(Agents(self.env_agents[pid], self.action_agents[pid])))
        #     self.acquisition_agents[pid].seed(kwargs["env_seed"])

        # Create workspace
        self.workspaces = [Workspace() for _ in range(self.n_particles)]

        # Setup optimizers
        optimizer_args = get_arguments(kwargs["optimizer"])
        self.optimizers = []
        for pid in range(self.n_particles):
            params = nn.Sequential(self.action_agents[pid], self.critic_agents[pid]).parameters()
            self.optimizers.append(get_class(
                kwargs["optimizer"](params, **optimizer_args)
            ))

    def execute_acquisition_agent(self, epoch):
        for pid in range(self.cfg.n_particles):
            if epoch > 0:
                self.workspaces[pid].zero_grad()
                self.workspaces[pid].copy_n_last_steps(1)
                self.acquisition_agents[pid](
                    self.workspace[pid], t=1, n_steps=self.n_steps - 1, stochastic=True
                )
            else:
                self.acquisition_agents[pid](self.workspace[pid], t=0, n_steps=self.n_steps, stochastic=True)

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

    def add_gradients(self, policy_loss, kernels):
        policy_loss.backward()

        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if j == i:
                    continue

                theta_i = self.action_agents[i].model.parameters()
                theta_j = self.action_agents[j].model.parameters()

                for (wi, wj) in zip(theta_i, theta_j):
                    wi.grad = wi.grad + wj.grad * kernels[j, i].detach()

    def compute_loss(self, alpha, logger, verbose=True):
        # Need to defined in inherited classes
        raise NotImplementedError

    def run_svpg(self, alpha, logger, show_losses=True, show_gradient=True):
        policy_loss, critic_loss = self.compute_loss(alpha, logger)
    

    


    

    

        

            