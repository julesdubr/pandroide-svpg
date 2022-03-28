import time

import torch
import hydra

from salina import Workspace
from salina.agents import Agents, NRemoteAgent, TemporalAgent

from algos.a2c.mono.agents import EnvAgent, create_a2c_agent, execute_agent
from algos.a2c.mono.loss import compute_a2c_loss, compute_critic_loss
from algos.a2c.mono.optimizer import setup_optimizers

from algos.svgd import *

from helpers.logger import Logger
from helpers.visu.visu_gradient import visu_loss_along_time


def compute_gradients_norms(particles, logger, epoch):
    policy_gradnorm, critic_gradnorm = 0, 0

    for particle in particles:

        prob_params = particle["prob_agent"].model.parameters()
        critic_params = particle["critic_agent"].critic_model.parameters()

        for w_prob, w_critic in zip(prob_params, critic_params):
            if w_prob.grad != None:
                policy_gradnorm += w_prob.grad.detach().data.norm(2) ** 2

            if w_critic.grad != None:
                critic_gradnorm += w_critic.grad.detach().data.norm(2) ** 2

    policy_gradnorm, critic_gradnorm = (
        torch.sqrt(policy_gradnorm),
        torch.sqrt(critic_gradnorm),
    )

    logger.add_log("Policy Gradient norm", policy_gradnorm, epoch)
    logger.add_log("Critic Gradient norm", critic_gradnorm, epoch)


def combine_agents(cfg, particles):
    # Combine all acquisition agent of all particle in a unique TemporalAgent.
    # This will help us to avoid using a loop explicitly to execute all these agents
    # (these agents will still be executed by a for loop by SaliNa)
    acq_agents = TemporalAgent(
        Agents(*[particle["acq_agent"] for particle in particles])
    )

    # Create the remote acquisition agent and the remote acquisition workspace
    acq_remote_agents, acq_workspace = NRemoteAgent.create(
        acq_agents,
        num_processes=cfg.algorithm.n_processes,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        stochastic=True,
    )
    # Set the seed
    acq_remote_agents.seed(cfg.algorithm.env_seed)

    # Combine all prob_agent of each particle to calculate the gradient
    prob_agents = Agents(*[particle["prob_agent"] for particle in particles])

    # We also combine all the critic_agent of all particle into a unique TemporalAgent
    tcritic_agent = TemporalAgent(
        Agents(*[particle["critic_agent"] for particle in particles])
    )

    return prob_agents, tcritic_agent, acq_remote_agents, acq_workspace


def create_particles(cfg, n_particles, env_agents):
    particles = list()
    for i in range(n_particles):
        # Create A2C agent for all particles
        acq_agent, prob_agent, critic_agent = create_a2c_agent(cfg, env_agents[i], i)
        particles.append(
            {
                "acq_agent": acq_agent,
                "prob_agent": prob_agent,
                "critic_agent": critic_agent,
            }
        )

    return particles


def get_parameters(nn_list):
    params = []
    for nn in nn_list:
        l = list(nn.parameters())

        l_flatten = [torch.flatten(p) for p in l]
        l_flatten = tuple(l_flatten)

        l_concat = torch.cat(l_flatten)

        params.append(l_concat)

    return torch.stack(params)


def compute_total_loss(cfg, particles, replay_workspace, alpha, logger, epoch, verbose):
    n_particles = len(particles)

    # Compute critic, entropy and a2c losses
    critic_loss, entropy_loss, a2c_loss = 0, 0, 0
    for i in range(n_particles):
        # Get relevant tensors (size are timestep * n_envs * ...)
        critic, done, action_probs, reward, action = replay_workspace[
            f"critic{i}",
            f"env{i}/done",
            f"action_probs{i}",
            f"env{i}/reward",
            f"action{i}",
        ]

        # Compute critic loss
        tmp, td = compute_critic_loss(cfg, reward, done, critic)
        critic_loss += tmp

        # Compute entropy loss
        entropy_loss += torch.distributions.Categorical(action_probs).entropy().mean()

        # Compute A2C loss
        a2c_loss -= (
            compute_a2c_loss(action_probs, action, td) * (1 / alpha) * (1 / n_particles)
        )

        # Compute the cumulated reward on final_state
        creward = replay_workspace[f"env{i}/cumulated_reward"]
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

    # Compute the first term in the SVGD update
    add_gradients(a2c_loss, kernels, particles, n_particles)

    loss = (
        -cfg.algorithm.entropy_coef * entropy_loss
        + cfg.algorithm.critic_coef * critic_loss
        # - cfg.algorithm.a2c_coef * a2c_loss
        + kernels.sum() / n_particles
    )

    return loss


def run_svpg(cfg, alpha=1, show_losses=False, show_gradients=False):
    # 1) Build the logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    n_particles = cfg.algorithm.n_particles
    env_agents = [EnvAgent(cfg, i) for i in range(n_particles)]

    # 3) Create the particles
    particles = create_particles(cfg, n_particles, env_agents)

    # 4) Combine the agents
    prob_agents, tcritic_agent, acq_remote_agents, acq_workspace = combine_agents(
        cfg, particles
    )

    # 5) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(
        cfg,
        [particle["prob_agent"] for particle in particles],
        [particle["critic_agent"] for particle in particles],
    )

    # 8) Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the remote acq_agent in the remote workspace
        execute_agent(cfg, epoch, acq_remote_agents, acq_workspace, particles)

        # Compute the prob and critic value over the whole replay workspace
        replay_workspace = Workspace(acq_workspace)
        prob_agents(replay_workspace, t=0, n_steps=cfg.algorithm.n_timesteps)
        tcritic_agent(replay_workspace, t=0, n_steps=cfg.algorithm.n_timesteps)

        # Sum up all the losses including the sum of kernel matrix and then use
        # backward() to automatically compute the gradient of the critic and the
        # second term in SVGD update
        loss = compute_total_loss(
            cfg, particles, replay_workspace, alpha, logger, epoch, show_losses
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the norm of gradient of the actor and gradient of the critic
        if show_gradients:
            compute_gradients_norms(particles, logger, epoch)


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
