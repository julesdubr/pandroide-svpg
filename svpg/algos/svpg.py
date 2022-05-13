import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, clip_grad_norm_

from salina.workspace import Workspace

from svpg.utils.utils import save_algo


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

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
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

    def run(self, save_dir, gamma=0.1, max_gradn=0.5, show_loss=False, show_grad=False):
        policy_loss = 0
        entropy_loss = 0
        tmp_epoch = 0
        steps = 0

        n_particles = self.algo.n_particles

        for epoch in range(self.algo.max_epochs):
            # Execute particles' agents
            self.algo.execute_train_agents(epoch)
            self.algo.execute_tcritic_agents()

            steps += self.algo.n_steps * self.algo.n_envs

            # Compute loss
            policy_loss, critic_loss, entropy_loss = self.algo.compute_loss(
                epoch, show_loss
            )

            critic_loss = self.algo.critic_coef * critic_loss / n_particles
            critic_loss.backward()

            if self.algo.clipped:
                for critic_agent in self.algo.critic_agents:
                    clip_grad_norm_(critic_agent.parameters(), max_gradn)

            # Critic gradient descent
            for critic_optimizer in self.algo.critic_optimizers:
                critic_optimizer.step()
            for critic_optimizer in self.algo.critic_optimizers:
                critic_optimizer.zero_grad()

            policy_loss = (
                policy_loss + self.algo.policy_coef * policy_loss / n_particles
            )
            entropy_loss = (
                entropy_loss + self.algo.entropy_coef * entropy_loss / n_particles
            )

            # Evaluation
            if epoch - tmp_epoch > self.algo.eval_interval:
                tmp_epoch = epoch
                self.algo.eval_timesteps.append(steps)

                if self.is_annealed:
                    gamma = self.annealed(steps)

                params = self.get_policy_parameters()
                params = params.to(self.algo.device)
                kernel = RBF()(params, params.detach())

                self.add_gradients(policy_loss * gamma / self.algo.n_particles, kernel)

                loss = entropy_loss + kernel.sum() / n_particles
                loss.backward()
                policy_loss = 0
                entropy_loss = 0

                if self.algo.clipped:
                    for action_agent in self.algo.action_agents:
                        clip_grad_norm_(action_agent.parameters(), max_gradn)

                # Actor radient descent
                for action_optimizer in self.algo.action_optimizers:
                    action_optimizer.step()
                for action_optimizer in self.algo.action_optimizers:
                    action_optimizer.zero_grad()

                # Log gradient norms
                if show_grad:
                    self.algo.compute_gradient_norm(epoch)

                for pid in range(self.algo.n_particles):
                    eval_workspace = Workspace()
                    self.algo.eval_agents[pid](
                        eval_workspace, t=0, stop_variable="env/done", stochastic=False
                    )
                    rewards = eval_workspace["env/cumulated_reward"][-1]
                    mean = rewards.mean()
                    self.algo.logger.add_log(f"reward_{pid}", mean, steps)
                    self.algo.rewards[pid].append(mean)

        ver = "SVPG_annealed" if self.is_annealed else "SVPG"
        save_algo(self.algo, save_dir, algo_version=ver)
