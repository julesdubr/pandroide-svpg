from salina import Workspace

from svpg.helpers.logger import Logger
from svpg.helpers.utils import compute_gradients_norms
from svpg.algos.a2c.mono.agents import execute_agent
from agents import EnvAgent, combine_agents
from particles import create_particles
from loss import compute_gradient
from optimizer import setup_optimizers

def run_svpg(cfg, alpha=10, show_losses=True, show_gradients=True):
    # 1) Build the logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    n_particles = cfg.algorithm.n_particles
    env_agents = [EnvAgent(cfg, i) for i in range(n_particles)]

    # 3) Create the particles
    particles = create_particles(cfg, n_particles, env_agents)

    # 4) Combine the agents
    acq_agent, tcritic_agent = combine_agents(
        cfg, particles
    )

    workspace = Workspace()

    # 5) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(
        cfg,
        [particle["prob_agent"] for particle in particles],
        [particle["critic_agent"] for particle in particles],
    )

    # 8) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the remote acq_agent in the remote workspace
        execute_agent(cfg, epoch, acq_agent, particles)
        # Sum up all the losses including the sum of kernel matrix and then use
        # backward() to automatically compute the gradient of the critic and the
        # second term in SVGD update
        compute_gradient(
            cfg, particles, workspace, logger, epoch, show_losses, alpha
        )
        optimizer.step()
        optimizer.zero_grad()

        # Compute the norm of gradient of the actor and gradient of the critic
        if show_gradients:
            compute_gradients_norms(particles, logger, epoch)