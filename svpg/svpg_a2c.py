import hydra
import time

from salina import Workspace
from salina import get_arguments, get_class

from svpg.helpers.logger import Logger

from svpg.algos.a2c_mono import A2CAlgoMono
from svpg.helpers.env import EnvAgent
from svpg.algos.svgd import *
from svpg.helpers.utils import *

from svpg.helpers.visu.visu_gradient import visu_loss_along_time


# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, prob_agents, critic_agents):
    optimizer_args = get_arguments(cfg.algorithm.optimizer)

    parameters = []
    for prob_agent, critic_agent in zip(prob_agents, critic_agents):
        parameters += list(prob_agent.parameters()) + list(critic_agent.parameters())

    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
    return optimizer


def compute_gradient(
    cfg,
    algo,
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
    critic_loss, entropy_loss, algo_loss = algo.compute_losses(
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


def run_svpg(cfg, alpha=10, show_loss=False, show_grad=False):
    algo = A2CAlgoMono()

    # 1) Build the logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    n_particles = cfg.algorithm.n_particles
    env_agents = [EnvAgent(cfg, i) for i in range(n_particles)]

    # 3) Create the particles
    acq_agents, prob_agents, critic_agents = algo.create_particles(
        cfg, n_particles, env_agents
    )

    # 4) Combine the agents
    tacq_agent, tcritic_agent = algo.combine_agents(cfg, acq_agents, critic_agents)

    workspace = Workspace()

    # 5) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, prob_agents, critic_agents)

    # 8) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the acq_agent in the workspace
        algo.execute_agent(cfg, epoch, workspace, tacq_agent)
        tcritic_agent(workspace, n_steps=cfg.algorithm.n_timesteps)
        # Sum up all the losses including the sum of kernel matrix and then use
        # backward() to automatically compute the gradient of the critic and the
        # second term in SVGD update
        compute_gradient(
            cfg,
            algo,
            prob_agents,
            critic_agents,
            workspace,
            logger,
            epoch,
            alpha,
            show_loss,
            show_grad,
        )

        optimizer.step()
        optimizer.zero_grad()


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    duration = time.process_time()
    losses, epoch = run_svpg(cfg)
    duration = time.process_time() - duration

    visu_loss_along_time(range(epoch + 1), losses, "loss_along_time")

    print(f"terminated in {duration}s at epoch {epoch}")


if __name__ == "__main__":
    main()
