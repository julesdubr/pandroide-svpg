from svpg.algos.a2c.mono.loss import *
from svpg.algos.svgd import *
from svpg.helpers.utils import *


def compute_gradient(
    cfg,
    prob_agents,
    critic_agents,
    workspace,
    logger,
    epoch,
    alpha=10,
    show_loss=True,
    show_grad=True,
):
    n_particles = len(prob_agents)

    # Compute critic, entropy and a2c losses
    critic_loss, entropy_loss, algo_loss = compute_losses(
        cfg, workspace, n_particles, epoch, logger, alpha
    )

    # Get the params
    params = get_parameters([prob_agents[i].model for i in range(n_particles)])

    # We need to detach the second list of params out of the computation graph
    # because we don't want to compute its gradient two time when using backward()
    kernels = RBF()(params, params.detach())

    loss = (
        -cfg.algorithm.entropy_coef * sum(entropy_loss)
        + cfg.algorithm.critic_coef * sum(critic_loss)
        + kernels.sum() / n_particles
    )

    # Compute the first term in SVGD update
    add_gradients(-sum(algo_loss), kernels, prob_agents, n_particles)

    # Compute the TD gradient as well as the seconde term in the SVGD update
    loss.backward()

    if show_grad:
        compute_gradients_norms(prob_agents, critic_agents, logger, epoch)
