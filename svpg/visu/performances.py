import numpy as np

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from pathlib import Path

from svpg.utils import load_algo
from salina.visu.common import final_show


def format_num(num, pos):
    magnitude = 0
    labels = ["", "K", "M", "G"]
    while abs(num) >= 1e3:
        magnitude += 1
        num /= 1e3

    return f"{num:.1f}{labels[magnitude]}"


def plot_algos_performances(
    directory, mode="mean", suffix="", save_fig=True, save_dir="../plots"
):
    _, ax = plt.subplots(figsize=(9, 6))
    formatter = FuncFormatter(format_num)

    colors = ["#09b542", "#008fd5", "#fc4f30", "#e5ae38", "#e5ae38", "#810f7c"]

    prefix = (Path(directory).parent.name + Path(directory).name).replace("-", "")
    env_name = Path(directory).parents[1].name
    algo_names = [path.name for path in Path(directory).iterdir() if path.is_dir()]

    for algo_name, color in zip(algo_names, colors):
        _, _, rewards, t = load_algo(directory + algo_name)

        if mode == "best":
            best = rewards.sum(axis=1).argmax()
            rewards = rewards[best]
        elif mode == "max":
            rewards = np.max(rewards, axis=0)
        else:
            std = rewards.std(axis=0)
            rewards = rewards.mean(axis=0)
            ax.fill_between(t, rewards + std, rewards - std, alpha=0.1, color=color)

        ax.plot(t, rewards, lw=2, label=f"{algo_name}", color=color)

    ax.xaxis.set_major_formatter(formatter)
    plt.legend()

    save_dir += f"/{env_name}"

    clean_env_name = env_name.split("-")[0]
    figname = f"/{prefix}_{clean_env_name.lower()}_{mode}"
    title = f"{clean_env_name} ({mode}"
    if suffix:
        figname += f"_{suffix}"
        title += f" {suffix}"
    title += ")"

    final_show(save_fig, True, figname, "timesteps", "rewards", title, save_dir)


def plot_histograms(
    rewards, env_name, suffix="", save_dir="../plots", plot=True, save_fig=True
):
    plt.figure(figsize=(9, 6))

    colors = ["#09b542", "#008fd5", "#fc4f30", "#e5ae38", "#e5ae38", "#810f7c"]
    # colors = ["#fc4f30", "#008fd5", "#e5ae38"]

    n_bars = len(rewards)
    x = np.arange(len(list(rewards.values())[0]))
    width = 0.75 / n_bars

    for i, reward in enumerate(rewards.values()):
        plt.bar(x + width * i, np.sort(reward)[::-1], width=width, color=colors[i])

    plt.legend(labels=rewards.keys())
    plt.xticks([], [])

    save_dir += f"/{env_name}"

    clean_env_name = env_name.split("-")[0]
    title = clean_env_name
    figname = f"/{clean_env_name.lower()}-histograms"

    if suffix:
        title += f" ({suffix})"
        figname += f"_{suffix}"

    final_show(save_fig, plot, figname, "", "rewards", title, save_dir)
