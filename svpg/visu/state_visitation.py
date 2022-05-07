from sklearn.manifold import TSNE

import seaborn as sns
import matplotlib.pyplot as plt

from salina.workspace import Workspace
from salina.agents import Agents, TemporalAgent

from svpg.agents.env import NoAutoResetEnvAgent

import os


def plot_state_visitation(
    cfg,
    agents,
    rewards,
    algo_name,
    directory="",
    nb_best=4,
    cmap="Reds",
    bw_adjust=0.75,
    plot=True,
    save=True,
    suptitle=True,
):
    rewards = rewards.mean(axis=1)
    bests_indices = rewards.argsort()[-nb_best:][::-1]

    tsne = TSNE(init="random", random_state=0, learning_rate="auto", n_iter=300)

    fig = plt.figure(figsize=(4 * nb_best, nb_best), constrained_layout=True)
    axes = fig.subplots(nrows=1, ncols=nb_best, sharey=True)

    for i, (pid, ax) in enumerate(zip(bests_indices, axes)):
        env_agent = NoAutoResetEnvAgent(cfg, n_envs=cfg.algorithm.n_evals)

        eval_agent = TemporalAgent(Agents(env_agent, agents[pid]))

        eval_workspace = Workspace()
        eval_agent(eval_workspace, t=0, stop_variable="env/done", stochastic=False)

        obs = eval_workspace["env/env_obs"]
        obs = obs.reshape(-1, obs.shape[2])

        Y = tsne.fit_transform(obs)

        sns.kdeplot(ax=ax, x=Y[:, 0], y=Y[:, 1], cmap=cmap, bw_adjust=bw_adjust)
        ax.axis("off")
        ax.set_title(f"#{i+1} ({round(rewards[pid])})")

    if suptitle:
        fig.suptitle(f"{algo_name} state visitation density", fontsize=14)

    if save:
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + f"{algo_name}_state_visitation.jpg"
        plt.savefig(filename)

    if plot:
        plt.show()
