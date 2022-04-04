import hydra
import torch

from salina import Workspace, instantiate_class

from svpg.algos.a2c.mono.agents import EnvAgent

from agents import create_particles
from loss import compute_losses
from optimizer import setup_optimizers


def compute_total_loss(cfg, entropy_loss, critic_loss, reinforce_loss):
    loss = (
        -cfg.algorithm.entropy_coef * entropy_loss
        + cfg.algorithm.critic_coef * critic_loss
        - cfg.algorithm.a2c_coef * reinforce_loss
    )
    return loss


def run_reinforce(cfg):
    logger = instantiate_class(cfg.logger)

    n_particles = cfg.algorithm.n_particles
    env_agents = [EnvAgent(cfg, i) for i in range(n_particles)]

    tacq_agent, reinforce_agents = create_particles(cfg, n_particles, env_agents)

    # 6) Configure the workspace to the right dimension. The time size is greater than
    # the naximum episode size to be able to store all episode states
    workspace = Workspace()

    # 7) Confgure the optimizer over the a2c agent
    optimizers = setup_optimizers(cfg, reinforce_agents)

    # 8) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):

        # Execute the agent on the workspace to sample complete episodes
        # Since not all the variables of workspace will be overwritten, it is better to clear the workspace
        workspace.clear()
        tacq_agent(workspace, stochastic=True, t=0, n_steps=cfg.algorithm.n_timesteps)
        # TODO: find a way to set the stop variables depending on the agent pid
        # tacq_agent(workspace, stochastic=True, t=0, stop_variable="env/done")

        critic_loss, entropy_loss, reinforce_loss = compute_losses(
            cfg, workspace, n_particles, epoch, logger
        )

        for i in range(n_particles):
            loss = compute_total_loss(
                cfg, i, critic_loss[i], entropy_loss[i], reinforce_loss[i]
            )
            optimizers[i].zero_grad()
            loss.backward()
            optimizers[i].step()


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_reinforce(cfg)


if __name__ == "__main__":
    main()