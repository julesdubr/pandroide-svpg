import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch as th


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
    :param directory: the directory where to save the file
    :return: nothing
    """
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if save_figure:
        directory = os.getcwd() + "./data/plots/" + directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + figure_name)
    if plot:
        plt.show()
    plt.close()


def plot_histograms(indep_rewards, svpg_rewards, title, save_figure=True, plot=True):
    x = np.arange(len(svpg_rewards))

    plt.bar(x + 0.1, np.sort(indep_rewards), width=0.2, color="red")
    plt.bar(x - 0.1, np.sort(svpg_rewards), width=0.2, color="blue")
    plt.legend(labels=[f"{title}-independent", f"{title}-SVPG"])

    final_show(
        save_figure, plot, f"{title}-indep_vs_svpg.pdf", "particules", "rewards", title
    )


def plot_pendulum(
    agent, env, plot=True, figname="pendulum_critic.pdf", save_figure=True
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
        raise (
            ValueError(
                "Observation space dimension {}, should be > 2".format(
                    env.observation_space.shape[0]
                )
            )
        )
    definition = 100
    portrait = np.zeros((definition, definition))
    state_min = env.observation_space.low
    state_max = env.observation_space.high

    for index_t, t in enumerate(np.linspace(-np.pi, np.pi, num=definition)):
        for index_td, td in enumerate(
            np.linspace(state_min[2], state_max[2], num=definition)
        ):
            obs = np.array([[np.cos(t), np.sin(t), td]])
            with th.no_grad():
                obs = th.from_numpy(obs.astype(np.float32))
                value = agent.model(obs).squeeze(-1)

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
        "Critic phase portrait",
    )


def plot_cartpole(
    agent,
    env,
    plot=True,
    figname="cartpole_critic.pdf",
    save_figure=True,
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
        raise (
            ValueError(
                "Observation space dimension {}, should be > 2".format(
                    env.observation_space.shape[0]
                )
            )
        )
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
            # Add batch dim
            obs = obs.reshape(1, -1)
            with th.no_grad():
                obs = th.from_numpy(obs.astype(np.float32))
                value = agent.model(obs).squeeze(-1)

            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[2], state_max[2]],
        aspect="auto",
    )
    plt.colorbar(label="critic value")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "V Function")
