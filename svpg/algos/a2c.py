import numpy as np

import salina.rl.functional as RLF

from svpg.algos.algo import Algo

import torch


class A2C(Algo):
    def __init__(
        self,
        policy_coef,
        critic_coef,
        entropy_coef,
        gae_coef,
        n_particles,
        max_epochs,
        discount_factor,
        env_name,
        max_episode_steps,
        n_envs,
        env_seed,
        eval_interval,
        clipped,
        n_steps,
        logger,
        env_agent,
        env,
        model,
        optimizer,
    ):
        super().__init__(
            n_particles,
            max_epochs,
            discount_factor,
            env_name,
            max_episode_steps,
            n_envs,
            env_seed,
            eval_interval,
            clipped,
            logger,
            env_agent,
            env,
            model,
            optimizer,
        )

        self.policy_coef, self.critic_coef, self.entropy_coef = (
            policy_coef,
            critic_coef,
            entropy_coef,
        )
        self.gae = gae_coef
        self.n_steps = n_steps
        self.T = n_steps * n_envs * max_epochs

    def compute_critic_loss(self, reward, must_bootstrap, critic):
        # Compute TD error
        # td = RLF.gae(critic, reward, must_bootstrap, self.discount_factor, self.gae)
        target = reward[:-1] + self.discount_factor * critic[1:].detach() * (must_bootstrap.float())
        td = target - critic
        # Compute critic loss
        td_error = td ** 2
        critic_loss = td_error.mean()

        return critic_loss, td

    def compute_policy_loss(self, action_logprobs, td):
        policy_loss = action_logprobs[:-1] * td.detach()

        return policy_loss.mean()

    def compute_loss(self, epoch, verbose=True):
        total_critic_loss, total_entropy_loss, total_policy_loss = 0, 0, 0

        for pid in range(self.n_particles):
            # Extracting the relevant tensors from the workspace
            transition_workspace = self.workspaces[pid].get_transitions()
            critic, done, action_logprobs, reward, entropy, truncated = transition_workspace[
                "critic", "env/done", "action_logprobs", "env/reward", "entropy", "env/truncated"
            ]

            # Move to gpu
            critic = critic.to(self.device)
            done = done.to(self.device)
            action_logprobs = action_logprobs.to(self.device)
            reward = reward.to(self.device)
            entropy = entropy.to(self.device)
            truncated = truncated.to(self.device)

            must_bootstrap = torch.logical_or(~done[1], truncated[1])

            # Compute loss
            critic_loss, td = self.compute_critic_loss(reward, must_bootstrap, critic)
            total_critic_loss = total_critic_loss + critic_loss

            total_entropy_loss = total_entropy_loss - entropy.mean()

            total_policy_loss = total_policy_loss - self.compute_policy_loss(
                action_logprobs, td
            )

        if verbose:
            self.logger.log_losses(
                epoch, total_critic_loss, total_entropy_loss, total_policy_loss
            )

        n_steps = np.full(self.n_particles, self.n_steps * self.n_env)
        return total_policy_loss, total_critic_loss, total_entropy_loss, n_steps
