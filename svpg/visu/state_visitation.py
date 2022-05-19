from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt

from salina.workspace import Workspace
from salina.agents import Agents, TemporalAgent

from svpg.agents.env import NoAutoResetEnvAgent
from svpg.utils.utils import load_algo

from omegaconf import OmegaConf
from pathlib import Path
import os


def get_embedded_spaces(directory, algo_name, nb_best=4, n_eval=100, seed=432):
    env_name = str(Path(directory).parents[1].name)

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

    agents, _, rewards, _ = load_algo(directory + algo_name)

    rewards = rewards.mean(axis=1)
    bests_indices = rewards.argsort()[-nb_best:][::-1]

    best_rewards = rewards[bests_indices]

    tsne = TSNE(init="random", random_state=0, learning_rate="auto", n_iter=300)

    outputs = []

    for pid in bests_indices:
        env_agent = NoAutoResetEnvAgent(config, n_envs=config.algorithm.n_evals)

        eval_agent = TemporalAgent(Agents(env_agent, agents[pid]))

        eval_workspace = Workspace()
        eval_agent(eval_workspace, t=0, stop_variable="env/done", stochastic=False)

        obs = eval_workspace["env/env_obs"]
        obs = obs.reshape(-1, obs.shape[2])

        outputs.append(tsne.fit_transform(obs))

    return outputs, best_rewards


def plot_state_visitation(
    directory,
    embedded_spaces,
    rewards,
    algo_name,
    suptitle=True,
    cmap="Blues",
    bw_adjust=0.75,
    save_fig=True,
    save_dir="../plots",
    plot=True,
):
    env_name = str(Path(directory).parents[1].name)

    n = rewards.shape[0]

    fig = plt.figure(figsize=(4 * n, n), constrained_layout=True)
    axes = fig.subplots(nrows=1, ncols=n, sharey=True)

    for i, ax in enumerate(axes):
        Y = embedded_spaces[i]

        sns.kdeplot(ax=ax, x=Y[:, 0], y=Y[:, 1], cmap=cmap, bw_adjust=bw_adjust)
        ax.axis("off")
        ax.set_title(f"#{i+1} ({round(rewards[i])})")

    clean_env_name = env_name.split("-")[0]

    if suptitle:
        fig.suptitle(
            f"({clean_env_name}) {algo_name} state visitation density", fontsize=14
        )

    if save_fig:
        save_dir += f"/{env_name}"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = save_dir + f"/{algo_name}_{clean_env_name.lower()}_SVD.jpg"
        plt.savefig(filename)

    if plot:
        plt.show()
