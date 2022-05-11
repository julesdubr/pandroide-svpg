from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt

from salina.workspace import Workspace
from salina.agents import Agents, TemporalAgent

from svpg.agents.env import NoAutoResetEnvAgent

import os


def get_embedded_spaces(cfg, agents, rewards, nb_best=4):
    rewards = rewards.sum(axis=1)
    bests_indices = rewards.argsort()[-nb_best:][::-1]

    best_rewards = rewards[bests_indices]

    tsne = TSNE(init="random", random_state=0, learning_rate="auto", n_iter=300)

    outputs = []

    for pid in bests_indices:
        env_agent = NoAutoResetEnvAgent(cfg, n_envs=cfg.algorithm.n_evals)

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
    suptitle=False,
    cmap="Blues",
    bw_adjust=0.75,
    save=False,
    plot=True,
):

    n = rewards.shape[0]

    fig = plt.figure(figsize=(4 * n, n), constrained_layout=True)
    axes = fig.subplots(nrows=1, ncols=n, sharey=True)

    for i, ax in enumerate(axes):
        Y = embedded_spaces[i]

        sns.kdeplot(ax=ax, x=Y[:, 0], y=Y[:, 1], cmap=cmap, bw_adjust=bw_adjust)
        ax.axis("off")
        ax.set_title(f"#{i+1} ({round(rewards[i])})")

    if suptitle:
        fig.suptitle(f"{algo_name} state visitation density", fontsize=14)

    if save:
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + f"{algo_name}_state_visitation.jpg"
        plt.savefig(filename)

    if plot:
        plt.show()
