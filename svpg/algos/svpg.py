from email import policy
import torch as th
from torch.nn.utils import parameters_to_vector

from itertools import permutations

from svpg.common.kernel import RBF


class SVPG():
    def __init__(self, algo):
        self.algo = algo
        self.kernel = RBF

    def get_policy_parameters(self):
        policy_params = [
            parameters_to_vector(action_agent.parameters())
            for action_agent in self.algo.action_agents
        ]
        return th.stack(policy_params)

    def add_gradients(self, policy_loss, kernel):
        policy_loss.backward(retain_graph=True)

        # Get all the couples of particules (i,j) st. i /= j
        # for i, j in list(permutations(range(self.algo.n_particles), r=2)):
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i == j:
                    continue

                theta_i = self.algo.action_agents[i].parameters()
                theta_j = self.algo.action_agents[j].parameters()

                for (wi, wj) in zip(theta_i, theta_j):
                    wi.grad = wi.grad + wj.grad * kernel[j, i].detach()

    def run(self, alpha=10, show_loss=True, show_grad=True):
        for epoch in range(self.algo.max_epochs):
            # Execute particles' agents
            self.algo.execute_acquisition_agent(epoch)
            self.algo.execute_critic_agent()

            # Compute loss
            policy_loss, critic_loss, entropy_loss, rewards = self.algo.compute_loss(epoch, show_loss)

            # Compute gradients
            params = self.get_policy_parameters()
            kernel = self.kernel()(params, params.detach())
            self.add_gradients(policy_loss * (1 / alpha) * (1 / self.algo.n_particles), kernel)

            loss = (
                + self.algo.entropy_coef * entropy_loss
                + self.algo.critic_coef * critic_loss
                + kernel.sum() / self.algo.n_particles
            )

            # Gradient descent
            for optimizer in self.optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in self.optimizers:
                optimizer.step()

            # Log gradient norms
            if show_grad:
                self.algo.compute_gradient_norm(epoch)
                
        return rewards
