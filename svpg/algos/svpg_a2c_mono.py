import torch

from svpg.algos.algo import Algo
from svpg.utils import _index


class SVPG_A2C_Mono(Algo):
    def __init__(self, cfg):
        super().__init__(cfg)

    def compute_critic_loss(self, reward, done, critic):
        # Compute TD error
        target = reward[1:] + self.discount_factor * critic[1:].detach() * (
            1 - done[1:].float()
        )
        td = target - critic[:-1]

        # Compute critic loss
        td_error = td ** 2
        critic_loss = td_error.mean()

        return critic_loss, td

    def compute_policy_loss(self, action_probs, action, td):
        action_logp = _index(action_probs, action).log()
        policy_loss = action_logp[:-1] * td.detach()

        return policy_loss.mean()

    def compute_loss(self, epoch, alpha=10, verbose=True):
        total_critic_loss, total_entropy_loss, total_policy_loss = 0, 0, 0
        for pid in range(self.n_particles):
            # Extracting the relevant tensors from the workspace
            critic, done, action_probs, reward, action = self.workspaces[pid][
                "critic", "env/done", "action_probs", "env/reward", "action"
            ]

            # Compute loss
            critic_loss, td = self.compute_critic_loss(reward, done, critic)
            total_critic_loss = total_critic_loss + critic_loss

            total_entropy_loss = (
                total_entropy_loss
                + torch.distributions.Categorical(action_probs).entropy().mean()
            )

            total_policy_loss = total_policy_loss - self.compute_policy_loss(
                action_probs, action, td
            ) * (1 / alpha) * (1 / self.n_particles)

            # Log reward
            creward = self.workspaces[pid]["env/cumulated_reward"]
            creward = creward[done]

            if creward.size()[0] > 0 and verbose:
                self.logger.add_log(f"reward_{pid}", creward.mean(), epoch)

        if verbose:
            self.logger.log_losses(
                epoch, total_critic_loss, total_entropy_loss, total_policy_loss
            )

        return total_critic_loss, total_entropy_loss, total_policy_loss
