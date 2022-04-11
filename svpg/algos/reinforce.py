from svpg.algos.algo import Algo

import torch as th
import numpy as np


class REINFORCE(Algo):
    def __init__(self,
                 policy_coef, critic_coef, 
                 n_particles, 
                 max_epochs, discount_factor,
                 env_name, max_episode_steps, n_envs, env_seed,
                 logger,
                 env_agent,
                 env, 
                 model, 
                 optimizer):
        super().__init__(n_particles, 
                        max_epochs, discount_factor,
                        env_name, max_episode_steps, n_envs, env_seed,
                        logger,
                        env_agent,
                        env, 
                        model, 
                        optimizer)

        self.policy_coef, self.critic_coef, self.entropy_coef = policy_coef, critic_coef, 0
        self.stop_variable = "env/done"

    def compute_reinforce_loss(self, reward, action_logprobs, critic, done):
        batch_size = reward.size()[1]  # Number of env
        max_trajectories_length = reward.size()[0]  # Longest episode over all env
        v_done, trajectories_length = done.float().max(0)
        trajectories_length += 1  # Episode length of each env
        assert v_done.eq(1.0).all()

        # Create mask to mask useless values (values that are enregistred into the
        # workspace after the episode end)
        # Create a matrix arange with size (max_trajectories_length, number of env).
        # arange(i, j) = i with 0 < i < max_trajectories_length
        arange = (
            th.arange(max_trajectories_length, device=done.device)
            .unsqueeze(-1)
            .repeat(1, batch_size)
        )
        # mask is a matrix with size (max_trajectories_length, number of env)
        # mask(i, j) = 1 if i < episode length in environment j, 0 otherwise
        mask = arange.lt(
            trajectories_length.unsqueeze(0).repeat(max_trajectories_length, 1)
        )

        # Mask useless reward
        reward = reward * mask

        # Compute discounted cumulated reward
        # cumulated_reward[t] = reward[t] + discount_factor * reward[t+1]
        #   + ... + (discount_factor ^ episode_length) * reward[episode_length]
        cumulated_reward = [th.zeros_like(reward[-1])]
        for t in range(max_trajectories_length - 1, 0, -1):
            cumulated_reward.append(
                self.discount_factor * cumulated_reward[-1] + reward[t]
            )
        cumulated_reward.reverse()
        cumulated_reward = th.cat([c.unsqueeze(0) for c in cumulated_reward])

        # Critic loss
        critic_loss = (
            ((critic - cumulated_reward) ** 2) * mask
        ).mean()  # use the value function (critic) as a baseline

        # Policy loss
        policy_loss = action_logprobs * (cumulated_reward - critic).detach()
        policy_loss = policy_loss * mask
        policy_loss = policy_loss.mean()

        return policy_loss, critic_loss

    def compute_loss(self, epoch, verbose=True):
        total_critic_loss, total_entropy_loss, total_policy_loss = 0, 0, 0
        rewards = np.zeros(self.n_particles)

        for pid in range(self.n_particles):
            # Extracting the relevant tensors from the workspace
            critic, done, action_logprobs, reward = self.workspaces[pid][
                "critic", "env/done", "action_logprobs", "env/reward"
            ]
            # Compute loss by REINFORCE
            # (using the reward cumulated until the end of episode)
            policy_loss, critic_loss = self.compute_reinforce_loss(
                reward, action_logprobs, critic, done
            )
            total_critic_loss = total_critic_loss + critic_loss
            total_policy_loss = total_policy_loss - policy_loss

            # Log reward
            creward = self.workspaces[pid]["env/cumulated_reward"]
            creward = creward[done]

            rewards[pid] = creward.mean()

            if creward.size()[0] > 0:
                self.logger.add_log(f"reward_{pid}", rewards[pid], epoch)

        if verbose:
            self.logger.add_log("policy_loss", total_policy_loss, epoch)
            self.logger.add_log("critic_loss", total_critic_loss, epoch)


        return total_policy_loss, total_critic_loss, 0, rewards
    
