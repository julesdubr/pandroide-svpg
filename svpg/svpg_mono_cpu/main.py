from salina import Workspace

from svpg.helpers.logger import Logger
from svpg.helpers.utils import compute_gradients_norms
from svpg.algos.a2c.mono.agents import execute_agent
from svpg.svpg_mono_cpu.agents import EnvAgent, combine_agents
from svpg.svpg_mono_cpu.particles import create_particles
from svpg.svpg_mono_cpu.loss import compute_gradient
from svpg.svpg_mono_cpu.optimizer import setup_optimizers


def run_svpg(cfg, alpha=10, show_losses=True, show_gradients=True):
    # 1) Build the logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    n_particles = cfg.algorithm.n_particles
    env_agents = [EnvAgent(cfg, i) for i in range(n_particles)]

    # 3) Create the particles
    particles = create_particles(cfg, n_particles, env_agents)

    # 4) Combine the agents
    acq_agent, tcritic_agent = combine_agents(cfg, particles)

    workspace = Workspace()

    # 5) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(
        cfg,
        [particle["prob_agent"] for particle in particles],
        [particle["critic_agent"] for particle in particles],
    )

    # 8) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the acq_agent in the workspace
        execute_agent(cfg, epoch, workspace, acq_agent)
        tcritic_agent(workspace, n_steps=cfg.algorithm.n_timesteps)
        # Sum up all the losses including the sum of kernel matrix and then use
        # backward() to automatically compute the gradient of the critic and the
        # second term in SVGD update
        compute_gradient(
            cfg, particles, workspace, logger, epoch, show_losses, alpha
        )

        optimizer.step()
        optimizer.zero_grad()
