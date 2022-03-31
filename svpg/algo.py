from salina import instantiate_class
from salina.agents import TemporalAgent, Agents
from salina.workspace import Workspace

from agents import EnvAgent
from logger import Logger

class Algo():
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = Logger(cfg)

        self.action_agents = [instantiate_class(cfg.prob_agent) for _ in range(self.n_particles)]
        self.critic_agents = [instantiate_class(cfg.critic_agent) for _ in range(self.n_particles)]
        self.tcritic_agents = [TemporalAgent(critic_agent) for critic_agent in self.critic_agents]
        self.env_agents = [EnvAgent(cfg) for _ in range(self.n_particles)]

        self.acquisition_agents = []
        for pid in range(cfg.algorithm.n_particles):
            self.acquisition_agents.append(TemporalAgent(Agents(self.env_agents[pid], self.action_agents[pid])))
            self.acquisition_agents[pid].seed(cfg.algorithm.env_seed)

        self.workspaces = [Workspace() for _ in range(cfg.algorithm.n_particles)]

    def execute_agent(self, epoch):
        for pid in range(self.cfg.n_particles):
            if epoch > 0:
                self.workspaces[pid].zero_grad()
                self.workspaces[pid].copy_n_last_steps(1)
                self.acquisition_agents[pid](
                    self.workspace[pid], t=1, n_steps=self.cfg.algorithm.n_timesteps - 1, stochastic=True
                )
            else:
                self.acquisition_agents[pid](self.workspace[pid], t=0, n_steps=self.cfg.algorithm.n_timesteps, stochastic=True)

    

    

        

            