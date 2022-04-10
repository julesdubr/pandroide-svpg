import torch
from svpg.algos.algo import Algo
from svpg.common.kernel import RBF


class SVPG(Algo):
    def __init__(self, cfg, algo):
        super().__init__(cfg)
        self.algo = algo
        self.kernel = RBF

    def get_policy_parameters(self):
        policy_params = []
        for pid in range(self.n_particles):
            l = list(self.algo.action_agents[pid].model.parameters())
            l_flatten = [torch.flatten(p) for p in l]
            l_flatten = tuple(l_flatten)
            l_concat = torch.cat(l_flatten)

            policy_params.append(l_concat)

        return torch.stack(policy_params)

    def add_gradients(self, policy_loss, kernel):
        policy_loss.backward(retain_graph=True)

        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if j == i:
                    continue

                theta_i = self.algo.action_agents[i].model.parameters()
                theta_j = self.algo.action_agents[j].model.parameters()

                for (wi, wj) in zip(theta_i, theta_j):
                    wi.grad = wi.grad + wj.grad * kernel[j, i].detach()

    def run(self, alpha=10, show_loss=False, show_grad=False):
        for epoch in range(self.max_epochs):
            # Execute particles' agents
            self.execute_acquisition_agent(epoch)
            self.execute_critic_agent()

            # Compute loss
            critic_loss, entropy_loss, policy_loss, rewards = self.algo.compute_loss(
                epoch, alpha, show_loss
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
