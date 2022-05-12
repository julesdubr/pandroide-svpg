from salina.workspace import Workspace
from salina.agents import Agents, TemporalAgent

from svpg.agents.env import NoAutoResetEnvAgent

from omegaconf import OmegaConf
from pathlib import Path
from collections import defaultdict

from svpg.utils.utils import load_algo


def eval_agent(config, agent, save_render=False):
    env_agent = NoAutoResetEnvAgent(config, n_envs=config.algorithm.n_evals)
    eval_agent = TemporalAgent(Agents(env_agent, agent))
    eval_workspace = Workspace()

    eval_agent(
        eval_workspace,
        t=0,
        stop_variable="env/done",
        stochastic=False,
        save_render=save_render,
    )

    reward = eval_workspace["env/cumulated_reward"][-1]
    return reward.mean().item()


def eval_agents_from_dir(directory, n_eval=100, seed=432, save_render=False):
    env_name = str(Path(directory).parents[1].name)
    algo_names = [path.name for path in Path(directory).iterdir() if path.is_dir()]

    config = OmegaConf.create(
        {
            "algorithm": {
                "seed": seed,
                "n_evals": n_eval,
            },
            "gym_env": {
                "classname": "svpg.agents.env.make_gym_env",
                "env_name": env_name,
            },
        }
    )

    rewards = defaultdict(list)

    for algo_name in algo_names:
        agents, _, _, _ = load_algo(directory + algo_name)

        for agent in agents:
            rewards[algo_name].append(
                eval_agent(config, agent, save_render=save_render)
            )

    return rewards
