import hydra
import time

from salina import Workspace

from svpg.helpers.logger import Logger
from svpg.algos.a2c.mono.agents import (
    execute_agent,
    EnvAgent,
    combine_agents,
    create_particles,
)

from loss import compute_gradient
from optimizer import setup_optimizers

from svpg.helpers.visu.visu_gradient import visu_loss_along_time


def run_svpg(cfg, alpha=10, show_losses=False, show_gradients=False):
    # 1) Build the logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    n_particles = cfg.algorithm.n_particles
    env_agents = [EnvAgent(cfg, i) for i in range(n_particles)]

    # 3) Create the particles
    acq_agents, prob_agents, critic_agents = create_particles(
        cfg, n_particles, env_agents
    )

    # 4) Combine the agents
    tacq_agent, tcritic_agent = combine_agents(cfg, acq_agents, critic_agents)

    workspace = Workspace()

    # 5) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, prob_agents, critic_agents)

    # 8) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the acq_agent in the workspace
        execute_agent(cfg, epoch, workspace, tacq_agent)
        tcritic_agent(workspace, n_steps=cfg.algorithm.n_timesteps)
        # Sum up all the losses including the sum of kernel matrix and then use
        # backward() to automatically compute the gradient of the critic and the
        # second term in SVGD update
        compute_gradient(
            cfg,
            prob_agents,
            critic_agents,
            workspace,
            logger,
            epoch,
            alpha,
            show_losses,
            show_gradients,
        )

        optimizer.step()
        optimizer.zero_grad()


@hydra.main(config_path="..", config_name="config.yaml")
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
