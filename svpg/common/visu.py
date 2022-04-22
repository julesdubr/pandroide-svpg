import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from pathlib import Path


def plot_algo_policies(algo, env, env_name, directory, plot=False):
    if "cartpole" in env_name.lower():
        plot_env = plot_cartpole
    elif "pendulum" in env_name.lower():
        plot_env = plot_pendulum
    else:
        print("Environment not supported for plot. Please use CartPole or Pendulum")
        return

    for pid in range(algo.n_particles):
        figname = f"policy_{pid}.png"
        plot_env(
            algo.action_agents[pid], env, figname, directory, plot, stochastic=True
        )

        figname = f"critic_{pid}.png"
        plot_env(algo.critic_agents[pid], env, figname, directory, plot)


def final_show(save_figure, plot, figure_name, x_label, y_label, title, directory):
    """
    Finalize all plots, adding labels and putting the corresponding file in the
    specified directory
    :param save_figure: boolean stating whether the figure should be saved
    :param plot: whether the plot should be shown interactively
    :param figure_name: the name of the file where to save the figure
    :param x_label: label on the x axis
    :param y_label: label on the y axis
    :param title: title of the figure
    :return: nothing
    """
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    if save_figure:
        if not os.path.exists(directory):
            os.makedirs(directory)
        directory = Path(directory + figure_name)
        plt.savefig(directory)

    if plot:
        plt.show()

    plt.close()


def plot_histograms(
    rewards_list, labels, colors, title, directory, plot=True, save_figure=True
):
    n_bars = len(rewards_list)
    x = np.arange(len(rewards_list[0]))
    width = 0.5 / n_bars

    for i, rewards in enumerate(rewards_list):
        plt.bar(x + width * i, np.sort(rewards)[::-1], width=width, color=colors[i])

    plt.legend(labels=labels)
    plt.xticks([], [])

    figname = f"{title}-indep_vs_svpg.png"
    final_show(save_figure, plot, figname, "", "rewards", title, directory)


def plot_pendulum(
    agent, env, figname, directory, plot=True, save_figure=True, stochastic=None
):
    """
    Plot a critic for the Pendulum environment
    :param agent: the policy / critic agent specifying the action to be plotted
    :param env: the evaluation environment
    :param plot: whether the plot should be interactive
    :param figname: the name of the file to save the figure
    :param save_figure: whether the figure should be saved
    :return: nothing
    """
    if env.observation_space.shape[0] <= 2:
        msg = f"Observation space dim {env.observation_space.shape[0]}, should be > 2"
        raise (ValueError(msg))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high

    for index_t, t in enumerate(np.linspace(-np.pi, np.pi, num=definition)):
        for index_td, td in enumerate(
            np.linspace(state_min[2], state_max[2], num=definition)
        ):
            obs = np.array([[np.cos(t), np.sin(t), td]])
            obs = th.from_numpy(obs.astype(np.float32))

            if stochastic is None:
                value = agent.model(obs).squeeze(-1)
            else:
                value = agent.forward(-1, stochastic, observation=obs)

            portrait[definition - (1 + index_td), index_t] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[-180, 180, state_min[2], state_max[2]],
        aspect="auto",
    )
    plt.colorbar(label="action")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(
        save_figure,
        plot,
        figname,
        x_label,
        y_label,
        "V Function",
        directory + "/pendulum_critics/",
    )


def plot_cartpole(
    agent, env, figname, directory, plot=True, save_figure=True, stochastic=None
):
    """
    Visualization of the critic in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first so as
    to plot them
    :param agent: the policy / critic agent to be plotted
    :param env: the environment
    :param plot: whether the plot should be interactive
    :param figname: the name of the file where to plot the function
    :param foldername: the name of the folder where to put the file
    :param save_figure: whether the plot should be saved into a file
    :return: nothing
    """
    if env.observation_space.shape[0] <= 2:
        msg = f"Observation space dim {env.observation_space.shape[0]}, should be > 2"
        raise (ValueError(msg))
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high

    for index_x, x in enumerate(
        np.linspace(state_min[0], state_max[0], num=definition)
    ):
        for index_y, y in enumerate(
            np.linspace(state_min[2], state_max[2], num=definition)
        ):
            obs = np.array([x])
            z1 = random.random() - 0.5
            z2 = random.random() - 0.5
            obs = np.append(obs, z1)
            obs = np.append(obs, y)
            obs = np.append(obs, z2)
            if stochastic is None:
                # Add batch dim
                obs = obs.reshape(1, -1)
                obs = th.from_numpy(obs.astype(np.float32))
                value = agent.model(obs).squeeze(-1)

            else:
                obs = th.from_numpy(obs.astype(np.float32))
                value = agent.forward(-1, stochastic, observation=obs)

            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[2], state_max[2]],
        aspect="auto",
    )

    if stochastic is None:
        directory += "/cartpole_critics/"
        title = "Cartpole Critic"
        plt.colorbar(label="critic value")
    else:
        directory += "/cartpole_policies/"
        title = "Cartpole Actor"
        plt.colorbar(label="action")

    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, title, directory)
