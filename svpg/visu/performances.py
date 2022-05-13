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

    env_name = str(Path(directory).parents[1].name)
    algo_names = [path.name for path in Path(directory).iterdir() if path.is_dir()]

    for algo_name in algo_names:
        _, _, rewards, timesteps = load_algo(directory + algo_name)

        if mode == "best":
            best = rewards.sum(axis=1).argmax()
            rewards = rewards[best]
        else:
            rewards = rewards.mean(axis=0)

        ax.plot(timesteps, rewards, linewidth=2, label=f"{algo_name}")

    ax.xaxis.set_major_formatter(formatter)
    plt.legend()

    save_dir += f"/{env_name}"

    clean_env_name = env_name.split("-")[0]
    figname = f"/{clean_env_name.lower()}_{mode}"
    title = f"{clean_env_name} ({mode}"
    if suffix:
        title += f" {suffix}"
    title += ")"

    if suffix:
        figname += f"_{suffix}"

    final_show(save_fig, True, figname, "timesteps", "rewards", title, save_dir)


def plot_histograms(
    rewards, env_name, suffix="", save_dir="../plots", plot=True, save_fig=True
):
    plt.figure(figsize=(9, 6))

    n_bars = len(rewards)
    x = np.arange(len(list(rewards.values())[0]))
    width = 0.5 / n_bars

    for i, reward in enumerate(rewards.values()):
        plt.bar(x + width * i, np.sort(reward)[::-1], width=width)

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