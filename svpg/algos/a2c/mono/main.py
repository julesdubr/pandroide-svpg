import hydra

from salina.workspace import Workspace

from svpg.helpers.logger import Logger

from svpg.algos.a2c.mono.agents import *
from svpg.algos.a2c.mono.loss import compute_losses
from svpg.algos.a2c.mono.optimizer import setup_optimizers


def run_a2c(cfg):
    """Main training loop of A2C"""

    # 1)  Build the  logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    n_particles = cfg.algorithm.n_particles
    env_agents = [EnvAgent(cfg, i) for i in range(n_particles)]

    # 3) Create the A2C Agent
    acq_agents, prob_agents, critic_agents = create_particles(
        cfg, n_particles, env_agents
    )

    # 4) Create the temporal critic agent to compute critic values over the workspace
    tacq_agent, tcritic_agent = combine_agents(cfg, acq_agents, critic_agents)

    # 5) Configure the workspace to the right dimension
    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the agent() and critic_agent()
    # will take the workspace as parameter
    workspace = Workspace()

    # 6) Configure the optimizer over the a2c agent
    optimizers = setup_optimizers(cfg, prob_agents, critic_agents)

    # 7) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        execute_agent(cfg, epoch, workspace, tacq_agent)

        # Compute the critic value over the whole workspace
        tcritic_agent(workspace, n_steps=cfg.algorithm.n_timesteps)

        losses = compute_losses(cfg, workspace, n_particles, epoch, logger)

        # Store the losses for tensorboard display
        # logger.log_losses(cfg, epoch, critic_loss, entropy_loss, a2c_loss)

        for optimizer, loss in zip(optimizers, losses):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    epoch, duration = run_a2c(cfg)

    print(f"terminated in {duration}s at epoch {epoch}")


if __name__ == "__main__":
    main()
