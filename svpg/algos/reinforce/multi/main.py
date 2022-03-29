import copy
import time

import hydra
import torch

from salina import Workspace, get_arguments, get_class, instantiate_class
from salina.agents import Agents, TemporalAgent
from salina.agents.gyma import AutoResetGymAgent
from salina.agents.remote import NRemoteAgent

from agents import REINFORCEAgent
from loss import compute_reinforce_loss


def run_reinforce(cfg):
    logger = instantiate_class(cfg.logger)

    # Get info on the environment
    env = instantiate_class(cfg.algorithm.env)
    observation_size = env.observation_space.shape[0]
    n_actions = env.action_space.n
    del env

    assert cfg.algorithm.n_envs % cfg.algorithm.n_processes == 0

    acq_env_agent = AutoResetGymAgent(
        get_class(cfg.algorithm.env),
        get_arguments(cfg.algorithm.env),
        n_envs=int(cfg.algorithm.n_envs / cfg.algorithm.n_processes),
    )

    r_agent = REINFORCEAgent(
        observation_size, cfg.algorithm.architecture.hidden_size, n_actions
    )

    acq_r_agent = copy.deepcopy(r_agent)
    acq_agent = TemporalAgent(Agents(acq_env_agent, acq_r_agent))
    acq_remote_agent, acq_workspace = NRemoteAgent.create(
        acq_agent,
        num_processes=cfg.algorithm.n_processes,
        t=0,
        n_steps=cfg.algorithm.n_timesteps,
        stochastic=True,
    )
    acq_remote_agent.seed(cfg.algorithm.env_seed)

    # 7) Confgure the optimizer over the a2c agent
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    optimizer = get_class(cfg.algorithm.optimizer)(
        r_agent.parameters(), **optimizer_args
    )

    # 8) Training loop
    epoch = 0
    for epoch in range(cfg.algorithm.max_epochs):
        pagent = acq_remote_agent.get_by_name("r_agent")
        for a in pagent:
            a.load_state_dict(r_agent.state_dict())

        # Execute the agent on the workspace to sample complete episodes
        # Since not all the variables of workspace will be overwritten, it is better to
        # clear the workspace
        replay_workspace = Workspace(acq_workspace)
        # acq_workspace.clear()
        acq_remote_agent(acq_workspace, stochastic=True, t=0, stop_variable="env/done")

        # Get relevant tensors (size are timestep x n_envs x ....)
        baseline, done, action_probs, reward, action = replay_workspace[
            "baseline", "env/done", "action_probs", "env/reward", "action"
        ]
        r_loss = compute_reinforce_loss(
            reward, action_probs, baseline, action, done, cfg.algorithm.discount_factor
        )

        # Log losses
        [logger.add_scalar(k, v.item(), epoch) for k, v in r_loss.items()]

        loss = (
            -cfg.algorithm.entropy_coef * r_loss["entropy_loss"]
            + cfg.algorithm.baseline_coef * r_loss["baseline_loss"]
            - cfg.algorithm.reinforce_coef * r_loss["reinforce_loss"]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute the cumulated reward on final_state
        creward = replay_workspace["env/cumulated_reward"]
        tl = done.float().argmax(0)
        creward = creward[tl, torch.arange(creward.size()[1])]
        logger.add_scalar("reward", creward.mean().item(), epoch)


@hydra.main(config_path=".", config_name="main.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")
    run_reinforce(cfg)


if __name__ == "__main__":
    main()
