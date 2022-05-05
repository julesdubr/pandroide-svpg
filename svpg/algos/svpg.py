import torch
from torch.nn.utils import parameters_to_vector

from salina.workspace import Workspace

from svpg.common.kernel import RBF

import numpy as np
import os
from pathlib import Path


class SVPG:
    def __init__(self, algo, is_annealed=True, slope=5, p=10, C=4, mode=1):
        self.algo = algo
        self.kernel = RBF
        self.is_annealed = is_annealed
        self.slope = slope
        self.p = p
        self.C = C
        self.mode = mode

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
        self,
        save_dir,
        gamma=1,
        p=5,
        slope=1.7,
        max_grad_norm=0.5,
        show_loss=False,
        show_grad=False,
    ):
        self.algo.to_gpu()
        nb_steps = np.zeros(self.algo.n_particles)
        last_epoch = 0

        for epoch in range(self.algo.max_epochs):
            # Execute particles' agents
            self.algo.execute_acquisition_agent(epoch)
            self.algo.execute_critic_agent()

            # Compute loss
            policy_loss, critic_loss, entropy_loss, n_steps = self.algo.compute_loss(
                epoch, show_loss
            )

            if self.is_annealed:
                t = np.max(nb_steps)
                gamma = self.annealed(t)

            # Compute gradients
            params = self.get_policy_parameters()
            params = params.to(self.algo.device)

            kernel = self.kernel()(params, params.detach())

            self.add_gradients(
                policy_loss * gamma * (1 / self.algo.n_particles), kernel
            )

            loss = (
                +self.algo.entropy_coef * entropy_loss / self.algo.n_particles
                + self.algo.critic_coef * critic_loss / self.algo.n_particles
                + kernel.sum() / self.algo.n_particles
            )

            loss.backward()

            if self.algo.clipped:
                for pid in range(self.algo.n_particles):
                    torch.nn.utils.clip_grad_norm_(
                        self.algo.action_agents[pid].parameters(), max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.algo.critic_agents[pid].parameters(), max_grad_norm
                    )

            # Log gradient norms
            if show_grad:
                self.algo.compute_gradient_norm(epoch)

            for optimizer in self.algo.optimizers:
                optimizer.step()

            # Gradient descent
            for optimizer in self.algo.optimizers:
                optimizer.zero_grad()

            # Evaluation
            nb_steps += n_steps
            if epoch - last_epoch == self.algo.eval_interval - 1:
                for pid in range(self.algo.n_particles):
                    eval_workspace = Workspace()
                    self.algo.eval_acquisition_agents[pid](
                        eval_workspace, t=0, stop_variable="env/done", stochastic=False
                    )
                    creward, done = (
                        eval_workspace["env/cumulated_reward"],
                        eval_workspace["env/done"],
                    )
                    creward, done = creward.to(self.algo.device), done.to(
                        self.algo.device
                    )
                    tl = done.float().argmax(0)
                    creward = creward[tl, torch.arange(creward.size()[1])]
                    self.algo.logger.add_log(
                        f"reward_{pid}", creward.mean(), nb_steps[pid]
                    )
                    self.algo.rewards[pid].append(creward.mean())
                    self.algo.eval_time_steps[pid].append(nb_steps[pid])
                    self.algo.eval_epoch[pid].append(epoch)

                last_epoch = epoch

        save_dir = (
            Path(str(save_dir) + "/svpg_annealed")
            if self.is_annealed
            else Path(str(save_dir) + "/svpg_normal")
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.algo.save_all_agents(str(save_dir))

        reward_path = (
            Path(str(save_dir) + "/reward_svpg_annealed.npy")
            if self.is_annealed
            else Path(str(save_dir) + "/reward_svpg.npy")
        )
        rewards_np = np.array(
            [[r for r in agent_reward] for agent_reward in self.algo.rewards.values()]
        )
        with open(reward_path, "wb") as f:
            np.save(f, rewards_np)
