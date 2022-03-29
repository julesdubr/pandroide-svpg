import torch, torch.nn as nn

from svpg.algos.a2c.mono.loss import *
from svpg.algos.svgd import *
from svpg.helpers.utils import *
from svpg.algos.svgd import *

def compute_gradient(cfg, particles, workspace, logger, epoch, verbose=True, alpha=10):
    n_particles = len(particles)

    # Compute critic, entropy and a2c losses
    critic_loss, entropy_loss, a2c_loss = 0, 0, 0
    for i in range(n_particles):
        # Get relevant tensors (size are timestep * n_envs * ...)
        critic, done, action_probs, reward, action = workspace[
            f"critic{i}",
            f"env{i}/done",
            f"action_probs{i}",
            f"env{i}/reward",
            f"action{i}",
        ]

        # Compute critic loss
        tmp, td = compute_critic_loss(cfg, reward, done, critic)
        critic_loss = critic_loss + tmp

        # Compute entropy loss
        entropy_loss = entropy_loss + torch.distributions.Categorical(action_probs).entropy().mean()

        # Compute A2C loss
        a2c_loss = a2c_loss - (
            compute_a2c_loss(action_probs, action, td) * (1 / alpha) * (1 / n_particles)
        )

        # Compute the cumulated reward on final_state
        creward = workspace[f"env{i}/cumulated_reward"]
        creward = creward[done]

        if creward.size()[0] > 0:
            logger.add_log(f"reward{i}", creward.mean(), epoch)

    if verbose:
        logger.log_losses(
            cfg,
            epoch,
            critic_loss.detach().mean(),
            entropy_loss.detach().mean(),
            a2c_loss.detach().mean(),
        )

    # Get the params
    params = get_parameters(
        [particles[i]["prob_agent"].model for i in range(n_particles)]
    )

    # We need to detach the second list of params out of the computation graph
    # because we don't want to compute its gradient two time when using backward()
    kernels = RBF()(params, params.detach())

    loss = (
        -cfg.algorithm.entropy_coef * entropy_loss
        + cfg.algorithm.critic_coef * critic_loss
        # - cfg.algorithm.a2c_coef * a2c_loss
        + kernels.sum() / n_particles
    )

    # Compute the first term in SVGD update
    add_gradients(a2c_loss, kernels, particles, n_particles)
    # Compute the TD gradient as well as the seconde term in the SVGD update
    loss.backward()

    if verbose:
        compute_gradients_norms(particles, logger, epoch)