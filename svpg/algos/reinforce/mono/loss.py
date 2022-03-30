import torch

from svpg.helpers.utils import _index


def compute_reinforce_loss(
    reward, action_probabilities, baseline, action, done, discount_factor
):
    """This function computes the reinforce loss, considering that episodes may have different lengths."""
    batch_size = reward.size()[1]

    # Find the first done occurence for each episode
    v_done, trajectories_length = done.float().max(0)
    trajectories_length += 1
    # assert v_done.eq(1.0).all()
    max_trajectories_length = trajectories_length.max().item()

    # Shorten trajectories for accelerate computation
    reward = reward[:max_trajectories_length]
    action_probabilities = action_probabilities[:max_trajectories_length]
    baseline = baseline[:max_trajectories_length]
    action = action[:max_trajectories_length]

    # Create a binary mask to mask useless values (of size max_trajectories_length x batch_size)
    arange = (
        torch.arange(max_trajectories_length, device=done.device)
        .unsqueeze(-1)
        .repeat(1, batch_size)
    )
    mask = arange.lt(
        trajectories_length.unsqueeze(0).repeat(max_trajectories_length, 1)
    )
    reward = reward * mask

    # Compute discounted cumulated reward
    cumulated_reward = [torch.zeros_like(reward[-1])]
    for t in range(max_trajectories_length - 1, 0, -1):
        cumulated_reward.append(discount_factor + cumulated_reward[-1] + reward[t])
    cumulated_reward.reverse()
    cumulated_reward = torch.cat([c.unsqueeze(0) for c in cumulated_reward])

    # baseline loss
    g = baseline - cumulated_reward
    baseline_loss = (g) ** 2
    baseline_loss = (baseline_loss * mask).mean()

    # policy loss
    log_probabilities = _index(action_probabilities, action).log()
    policy_loss = log_probabilities * -g.detach()
    policy_loss = policy_loss * mask
    policy_loss = policy_loss.mean()

    # entropy loss
    entropy = torch.distributions.Categorical(action_probabilities).entropy() * mask
    entropy_loss = entropy.mean()

    return {
        "baseline_loss": baseline_loss,
        "reinforce_loss": policy_loss,
        "entropy_loss": entropy_loss,
    }


def compute_losses(cfg, workspace, n_particles, epoch, logger, alpha=None):
    baseline_loss, entropy_loss, reinforce_loss = list(), list(), list()

    const = 1
    if alpha != None:
        const /= alpha * n_particles

    for i in range(n_particles):
        baseline, done, action_probs, reward, action = workspace[
            f"baseline{i}",
            f"env{i}/done",
            f"action_probs{i}",
            f"env{i}/reward",
            f"action{i}",
        ]
        r_loss = compute_reinforce_loss(
            reward, action_probs, baseline, action, done, cfg.algorithm.discount_factor
        )

        baseline_loss.append(r_loss["baseline_loss"])
        entropy_loss.append(r_loss["entropy_loss"])
        reinforce_loss.append(r_loss["reinforce_loss"] * const)

        # Compute the cumulated reward on final_state
        creward = workspace[f"env{i}/cumulated_reward"]
        tl = done.float().argmax(0)
        creward = creward[tl, torch.arange(creward.size()[1])]
        logger.add_scalar(f"reward{i}", creward.mean().item(), epoch)

    return baseline_loss, entropy_loss, reinforce_loss
