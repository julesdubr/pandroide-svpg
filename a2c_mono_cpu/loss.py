from helper import _index


def compute_critic_loss(cfg, reward, done, critic):
    """
    Compute critic loss
    """

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
