import numpy as np

import salina.rl.functional as RLF

from svpg.algos.algo import Algo


class A2C(Algo):
    def __init__(self,
                 policy_coef, critic_coef, entropy_coef, gae_coef, 
                 n_particles, n_samples, 
                 max_epochs, discount_factor,
                 env_name, max_episode_steps, n_envs, env_seed,
                 n_steps,
                 logger,
                 env_agent,
                 env, 
                 model, 
                 optimizer):
        super().__init__(n_particles, n_samples, 
                        max_epochs, discount_factor,
                        env_name, max_episode_steps, n_envs, env_seed,
                        logger,
                        env_agent,
                        env, 
                        model, 
                        optimizer)

        self.policy_coef, self.critic_coef, self.entropy_coef = policy_coef, critic_coef, entropy_coef
        self.gae = gae_coef
        self.n_steps = n_steps

    def compute_critic_loss(self, reward, done, critic):
        # Compute TD error
        td = RLF.gae(critic, reward, done, self.discount_factor, self.gae)
        # Compute critic loss
        td_error = td ** 2

        print(td_error.size())

        critic_loss = td_error.mean()

        return critic_loss, td

    def compute_policy_loss(self, action_logprobs, td):
        policy_loss = action_logprobs[:-1] * td.detach()

        print(policy_loss.size())

        return policy_loss.mean()

    def compute_loss(self, epoch, verbose=True):
        total_critic_loss, total_entropy_loss, total_policy_loss = 0, 0, 0

        for pid in range(self.n_particles):
            # Extracting the relevant tensors from the workspace
            critic, done, action_logprobs, reward, entropy = self.workspaces[pid][
                "critic", "env/done", "action_logprobs", "env/reward", "entropy"
            ]

            # Compute loss
            critic_loss, td = self.compute_critic_loss(reward, done, critic)
            total_critic_loss = total_critic_loss + critic_loss

            total_entropy_loss = total_entropy_loss - entropy.mean()

            total_policy_loss = total_policy_loss - self.compute_policy_loss(
                action_logprobs, td
            )

            # Log reward
            creward = self.workspaces[pid]["env/cumulated_reward"]
            creward = creward[done]

            self.rewards[epoch, pid] = creward.mean()

            if creward.size()[0] > 0:
                self.logger.add_log(f"reward_{pid}", self.rewards[epoch, pid], epoch)

        if verbose:
            self.logger.log_losses(
                epoch, total_critic_loss, total_entropy_loss, total_policy_loss
            )

        n_samples = critic.size()[0] * critic.size()[1]

        return total_policy_loss, total_critic_loss, total_entropy_loss, n_samples
