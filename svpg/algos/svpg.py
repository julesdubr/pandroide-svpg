import torch as th
from torch.nn.utils import parameters_to_vector

from svpg.algos.algo import Algo

from itertools import permutations


class SVPG(Algo):
    def __init__(self, cfg, algo, kernel):
        super().__init__(cfg)
        self.algo = algo
        self.kernel = kernel

    def get_policy_parameters(self):
        policy_params = [
            parameters_to_vector(action_agent.model.parameters())
            for action_agent in range(self.action_agents)
        ]
        return th.stack(policy_params)

    def add_gradients(self, policy_loss, kernel):
        policy_loss.backward(retain_graph=True)

        # Get all the couples of particules (i,j) st. i /= j
        for i, j in list(permutations(range(self.n_particles), r=2)):

            theta_i = self.action_agents[i].model.parameters()
            theta_j = self.action_agents[j].model.parameters()

            for (wi, wj) in zip(theta_i, theta_j):
                wi.grad = wi.grad + wj.grad * kernel[j, i].detach()

    def run(self, alpha=10, show_loss=False, show_grad=False):
        for epoch in range(self.max_epochs):
            # Execute particles' agents
            self.execute_acquisition_agent(epoch)
            self.execute_critic_agent()

            # Compute loss
            critic_loss, entropy_loss, policy_loss, rewards = self.algo.compute_loss(
                self.workspaces, self.logger, epoch, alpha, show_loss
            )

            # Compute gradients
            params = self.get_policy_parameters()
            kernel = self.kernel()(params, params.detach())
            self.add_gradients(policy_loss, kernel)

            loss = (
                -self.entropy_coef * entropy_loss
                + self.critic_coef * critic_loss
                + kernel.sum() / self.n_particles
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
