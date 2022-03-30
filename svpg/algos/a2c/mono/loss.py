from svpg.helpers.utils import _index
import torch


def compute_critic_loss(cfg, reward, done, critic):
    """Compute critic loss"""

    # Compute temporal difference
    target = reward[1:] + cfg.algorithm.discount_factor * critic[1:].detach() * (
        1 - done[1:].float()
    )
    td = target - critic[:-1]

    # Compute critic loss
    td_error = td ** 2
    critic_loss = td_error.mean()

    return critic_loss, td


def compute_a2c_loss(action_probs, action, td):
    """
    Compute A2C loss
    """
    action_logp = _index(action_probs, action).log()
    a2c_loss = action_logp[:-1] * td.detach()
    return a2c_loss.mean()


def compute_losses(cfg, workspace, n_particles, epoch, logger):
    losses = list()

    for i in range(n_particles):
        # Get relevant tensors (size are timestep x n_envs x ....)
        critic, done, action_probs, reward, action = workspace[
            f"critic{i}",
            f"env{i}/done",
            f"action_probs{i}",
            f"env{i}/reward",
            f"action{i}",
        ]

        # Compute critic loss
        critic_loss, td = compute_critic_loss(cfg, reward, done, critic)

        # Compute entropy loss
        entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()

        # Compute A2C loss
        a2c_loss = compute_a2c_loss(action_probs, action, td)

        losses.append(
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )

        # Compute the cumulated reward on final_state
        creward = workspace[f"env{i}/cumulated_reward"]
        creward = creward[done]

        if creward.size()[0] > 0:
            logger.add_log(f"reward{i}", creward.mean(), epoch)

    return losses
