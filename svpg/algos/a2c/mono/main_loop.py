from svpg.helpers.logger import Logger

from svpg.algos.a2c.mono.agents import create_a2c_agent, execute_agent, EnvAgent
from svpg.algos.a2c.mono.loss import compute_critic_loss, compute_a2c_loss
from svpg.algos.a2c.mono.optimizer import setup_optimizers

import salina
import torch


def run_a2c(cfg):
    """
    Main training loop of A2C
    """

    # 1)  Build the  logger
    logger = Logger(cfg)

    # 2) Create the environment agent
    env_agent = EnvAgent(cfg)

    # 3) Create the A2C Agent
    a2c_agent, prob_agent, critic_agent = create_a2c_agent(cfg, env_agent)

    # 4) Create the temporal critic agent to compute critic values over the workspace
    tcritic_agent = salina.agents.TemporalAgent(critic_agent)

    # 5) Configure the workspace to the right dimension
    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the agent() and critic_agent()
    # will take the workspace as parameter
    workspace = salina.Workspace()

    # 6) Configure the optimizer over the a2c agent
    optimizer = setup_optimizers(cfg, prob_agent, critic_agent)

    # 7) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        execute_agent(cfg, epoch, workspace, a2c_agent)

        # Compute the critic value over the whole workspace
        tcritic_agent(workspace, n_steps=cfg.algorithm.n_timesteps)

        # Get relevant tensors (size are timestep x n_envs x ....)
        critic, done, action_probs, reward, action = workspace[
            "critic", "env/done", "action_probs", "env/reward", "action"
        ]

        # Compute critic loss
        critic_loss, td = compute_critic_loss(cfg, reward, done, critic)

        # Compute entropy loss
        entropy_loss = torch.distributions.Categorical(action_probs).entropy().mean()

        # Compute A2C loss
        a2c_loss = compute_a2c_loss(action_probs, action, td)

        # Store the losses for tensorboard display
        logger.log_losses(cfg, epoch, critic_loss, entropy_loss, a2c_loss)

        # Compute the total loss
        loss = (
            -cfg.algorithm.entropy_coef * entropy_loss
            + cfg.algorithm.critic_coef * critic_loss
            - cfg.algorithm.a2c_coef * a2c_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the cumulated reward on final_state
        creward = workspace["env/cumulated_reward"]
        creward = creward[done]
        if creward.size()[0] > 0:
            logger.add_log("reward", creward.mean(), epoch)
