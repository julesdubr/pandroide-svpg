import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, clip_grad_norm_

from salina.workspace import Workspace

from svpg.utils.utils import save_algo_data


class RBF(nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma**2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


class SVPG:
    def __init__(self, algo, is_annealed=True, slope=5, p=10, C=4, mode=1):
        self.algo = algo
        self.is_annealed = is_annealed
        self.mode = mode
        self.slope = slope
        self.p = p
        self.C = C

    def get_policy_parameters(self):
        policy_params = [
            parameters_to_vector(action_agent.parameters())
            for action_agent in self.algo.action_agents
        ]
        return torch.stack(policy_params)

    def add_gradients(self, policy_loss, kernel):
        policy_loss.backward(retain_graph=True)

        # Get all the couples of particules (i,j) st. i /= j
        # for i, j in list(permutations(range(self.algo.n_particles), r=2)):
        for i in range(self.algo.n_particles):
            for j in range(self.algo.n_particles):
                if i == j:
                    continue

                theta_i = self.algo.action_agents[i].parameters()
                theta_j = self.algo.action_agents[j].parameters()

                for (wi, wj) in zip(theta_i, theta_j):
                    wi.grad = wi.grad + wj.grad * kernel[j, i].detach()

    def annealed(self, t):
        if self.mode == 1:
            return np.tanh((self.slope * t / self.algo.T) ** self.p)

        elif self.mode == 2:
            mod = t % (self.algo.T / self.C)
            return (mod / (self.algo.T / self.C)) ** self.p

    def run(
        self, save_dir, gamma=1, max_grad_norm=0.5, show_loss=False, show_grad=False
    ):
        self.algo.to_device()

        nb_steps = 0
        tmp_steps = 0

        for epoch in range(self.algo.max_epochs):
            # Execute particles' agents
            self.algo.execute_train_agents(epoch)
            self.algo.execute_tcritic_agents()

            nb_steps += self.algo.n_steps * self.algo.n_envs

            # Compute loss
            policy_loss, critic_loss, entropy_loss = self.algo.compute_loss(
                epoch, show_loss
            )

            if self.is_annealed:
                t = np.max(nb_steps)
                gamma = self.annealed(t)

            # Compute gradients
            params = self.get_policy_parameters()
            params = params.to(self.algo.device)
            kernel = RBF()(params, params.detach())

            self.add_gradients(policy_loss * gamma / self.algo.n_particles, kernel)

            loss = (
                +self.algo.entropy_coef * entropy_loss / self.algo.n_particles
                + self.algo.critic_coef * critic_loss / self.algo.n_particles
                + kernel.sum() / self.algo.n_particles
            )

            if self.algo.clipped:
                for pid in range(self.algo.n_particles):
                    clip_grad_norm_(
                        self.algo.action_agents[pid].parameters(), max_grad_norm
                    )
                    clip_grad_norm_(
                        self.algo.critic_agents[pid].parameters(), max_grad_norm
                    )

            # Log gradient norms
            if show_grad:
                self.algo.compute_gradient_norm(epoch)

            # Gradient descent
            for optimizer in self.algo.optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in self.algo.optimizers:
                optimizer.step()

            # Evaluation
            if nb_steps - tmp_steps > self.algo.eval_interval:
                tmp_steps = nb_steps

                for pid in range(self.algo.n_particles):
                    eval_workspace = Workspace().to(self.algo.device)
                    self.algo.eval_agents[pid](
                        eval_workspace, t=0, stop_variable="env/done", stochastic=False
                    )
                    rewards = eval_workspace["env/cumulated_reward"][-1]
                    mean = rewards.mean()
                    self.algo.logger.add_log(f"reward_{pid}", mean, nb_steps)
                    self.algo.rewards[pid].append(mean)

        ver = "SVPG_annealed" if self.is_annealed else "SVPG"
        save_algo_data(self.algo, save_dir, algo_version=ver)
