import hydra
import torch

from salina import Workspace, get_arguments, get_class, instantiate_class
from salina.agents.gyma import GymAgent
from salina.agents import Agents, TemporalAgent

from svpg.algos.a2c.mono.agents import EnvAgent

from agents import create_particles
from loss import compute_losses
from optimizer import setup_optimizers


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
        # tacq_agent(workspace, stochastic=True, t=0, stop_variable="env{???}/done")

        losses = compute_losses(cfg, workspace, n_particles, epoch, logger)

        for optimizer, loss in zip(optimizers, losses):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_reinforce(cfg)


if __name__ == "__main__":
    main()
