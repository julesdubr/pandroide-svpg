import random

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from svpg.helpers.visu.visu_policies import final_show

from stable_baselines3 import DDPG, REINFORCE
from stable_baselines3.common.utils import obs_as_tensor


def plot_critic(simu, model, study, default_string, num):
    """
    The main entry point for plotting a critic: determine which plotting function to
    call depending on the environment parameters
    :param simu: the simulation, which contains information about the environment, obs_size...
    :param model: the policy and critic used
    :param study: the name of the current study
    :param default_string: a string used to further specify the file name
    :param num: a number to plot several critics from the same configuration
    :return: nothing
    """
    picname = str(num) + "_critic_" + study + default_string + simu.env_name + ".pdf"
    env = simu.env
    obs_size = simu.obs_size
    deterministic = True

    if not simu.discrete:
        if obs_size == 1:
            plot_func = plot_qfunction_1d
        elif obs_size == 2:
            plot_func = plot_2d_critic
        else:
            plot_func = plot_nd_critic
        plot_func(
            model,
            env,
            deterministic,
            plot=False,
            save_figure=True,
            figname=picname,
            foldername="/plots/",
        )


# Visualization of the V function for a 2D environment like continuous mountain car.
# The action does not matter.
def plot_2d_critic(
    model,
    env,
    plot=True,
    figname="vfunction.pdf",
    foldername="/plots/",
    save_figure=True,
) -> None:
    """
    Plot a value function in a 2-dimensional state space
    :param model: the policy and critic to be plotted
    :param env: the environment
    :param plot: whether the plot should be interactive
    :param figname: the name of the file where to plot the function
    :param foldername: the name of the folder where to put the file
    :param save_figure: whether the plot should be saved into a file
    :return: nothing
    """
    if env.observation_space.shape[0] != 2:
        raise (
            ValueError(
                "Observation space dimension {}, should be 2".format(
                    env.observation_space.shape[0]
                )
            )
        )

    definition = 100
    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            obs = np.array([[x, y]])
            with th.no_grad():
                if hasattr(model, "critic"):
                    # For REINFORCE
                    if isinstance(model, REINFORCE):
                        value = model.critic.forward(obs_as_tensor(obs, model.device))
                    elif isinstance(model, DDPG):
                        action = model.forward(obs_as_tensor(obs, model.device))
                        value = model.critic.forward(
                            obs_as_tensor(obs, model.device), action
                        )
                    else:
                        action = model.forward(obs_as_tensor(obs, model.device))
                        value = model.critic.forward(
                            obs_as_tensor(obs, model.device), action
                        )
                        print("visu critic: algo not covered")
                else:
                    # For A2C/PPO/DDPG
                    action = model.forward(obs_as_tensor(obs, model.device))
                    value = model.predict_values(
                        obs_as_tensor(obs, model.device), action
                    )
            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect="auto"
    )
    plt.colorbar(label="critic value")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "V Function", foldername)


def plot_nd_critic(
    model,
    env,
    plot=True,
    figname="vfunction.pdf",
    foldername="/plots/",
    save_figure=True,
) -> None:
    """
    Visualization of the critic in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first so as to plot them
    :param model: the policy and critic to be plotted
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
            np.linspace(state_min[1], state_max[1], num=definition)
        ):
            obs = np.array([[x, y]])
            for _ in range(2, len(state_min)):
                z = random.random() - 0.5
                obs = np.append(obs, z)
                with th.no_grad():
                    if hasattr(model, "critic"):
                        # For REINFORCE
                        if isinstance(model, REINFORCE):
                            value = model.critic.forward(
                                obs_as_tensor(obs, model.device)
                            )
                        elif isinstance(model, DDPG):
                            action = model.forward(obs_as_tensor(obs, model.device))
                            value = model.critic.forward(
                                obs_as_tensor(obs, model.device), action
                            )
                        else:
                            print("visu critic: algo not covered")
                    else:
                        # For A2C/PPO/DDPG
                        action = model.forward(obs_as_tensor(obs, model.device))
                        value = model.predict_values(
                            obs_as_tensor(obs, model.device), action
                        )
                portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait,
        cmap="inferno",
        extent=[state_min[0], state_max[0], state_min[1], state_max[1]],
        aspect="auto",
    )
    plt.colorbar(label="critic value")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "V Function", foldername)


# visualization of the Q function for a 1D environment like 1D Toy with continuous actions
def plot_qfunction_1d(
    model,
    env,
    plot=True,
    figname="qfunction_1D.pdf",
    foldername="/plots/",
    save_figure=True,
) -> None:
    """
    Plot a q function in a 1-dimensional state space. The second dimension covers the whole action space
    :param model: the policy and critic to be plotted
    :param env: the environment
    :param plot: whether the plot should be interactive
    :param figname: the name of the file where to plot the function
    :param foldername: the name of the folder where to put the file
    :param save_figure: whether the plot should be saved into a file
    :return: nothing
    """
    if env.observation_space.shape[0] != 1:
        raise (
            ValueError(
                "The observation space dimension is {}, should be 1".format(
                    env.observation_space.shape[0]
                )
            )
        )
    definition = 100
    portrait = np.zeros((definition, definition))
    x_min = env.observation_space.low[0]
    x_max = env.observation_space.high[0]
    y_min = env.action_space.low[0]
    y_max = env.action_space.high[0]

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, _ in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            obs = np.array([x])
            with th.no_grad():
                if hasattr(model, "critic"):
                    # For REINFORCE
                    if isinstance(model, REINFORCE):
                        value = model.critic.forward(obs_as_tensor(obs, model.device))
                    elif isinstance(model, DDPG):
                        action = model.forward(obs_as_tensor(obs, model.device))
                        value = model.critic.forward(
                            obs_as_tensor(obs, model.device), action
                        )
                    else:
                        print("visu critic: algo not covered")
                else:
                    # For A2C/PPO/DDPG
                    action = model.forward(obs_as_tensor(obs, model.device))
                    value = model.predict_values(
                        obs_as_tensor(obs, model.device), action
                    )
            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect="auto"
    )
    plt.colorbar(label="critic value")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(save_figure, plot, figname, x_label, y_label, "Q Function", foldername)


def plot_qfunction_cont_act(
    model,
    env,
    plot=True,
    figname="qfunction_cont.pdf",
    foldername="/plots/",
    save_figure=True,
) -> None:
    """
    Visualization of the Q function for a 2D environment like continuous mountain car.
    Uses the same action everywhere in the state space
    :param model: the policy and critic to be plotted
    :param env: the environment
    :param plot: whether the plot should be interactive
    :param figname: the name of the file where to plot the function
    :param foldername: the name of the folder where to put the file
    :param save_figure: whether the plot should be saved into a file
    :return: nothing
    """
    if env.observation_space.shape[0] < 2:
        raise (
            ValueError(
                "The observation space dimension is {}, whereas it should be 2".format(
                    env.observation_space.shape[0]
                )
            )
        )
    definition = 100
    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            obs = np.array([[x, y]])
            with th.no_grad():
                if hasattr(model, "critic"):
                    # For REINFORCE
                    if isinstance(model, REINFORCE):
                        value = model.critic.forward(obs_as_tensor(obs, model.device))
                    elif isinstance(model, DDPG):
                        action = model.forward(obs_as_tensor(obs, model.device))
                        value = model.critic.forward(
                            obs_as_tensor(obs, model.device), action
                        )
                    else:
                        print("visu critic: algo not covered")
                else:
                    # For A2C/PPO/DDPG
                    action = model.forward(obs_as_tensor(obs, model.device))
                    value = model.predict_values(
                        obs_as_tensor(obs, model.device), action
                    )
            portrait[definition - (1 + index_y), index_x] = value.item()

    plt.figure(figsize=(10, 10))
    plt.imshow(
        portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect="auto"
    )
    plt.colorbar(label="critic value")
    # Add a point at the center
    plt.scatter([0], [0])
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])
    final_show(
        save_figure,
        plot,
        figname,
        x_label,
        y_label,
        "Q Function or current policy",
        foldername,
    )


def plot_pendulum_critic(
    model, env, plot=True, figname="pendulum_critic.pdf", save_figure=True
) -> None:
    """
    Plot a critic for the Pendulum environment
    :param model: the policy and critic specifying the action to be plotted
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
                if hasattr(model, "critic"):
                    # For REINFORCE
                    if isinstance(model, REINFORCE):
                        value = model.critic.forward(obs_as_tensor(obs, model.device))
                    elif isinstance(model, DDPG):
                        action = model.forward(obs_as_tensor(obs, model.device))
                        value = model.critic.forward(
                            obs_as_tensor(obs, model.device), action
                        )
                    else:
                        print("visu critic: algo not covered")
                else:
                    # For A2C/PPO/DDPG
                    action = model.forward(obs_as_tensor(obs, model.device))
                    value = model.predict_values(
                        obs_as_tensor(obs, model.device), action
                    )
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
        save_figure, plot, figname, x_label, y_label, "Critic phase portrait", "/plots/"
    )


def plot_cartpole_critic(
    model,
    env,
    plot=True,
    figname="cartpole_critic.pdf",
    foldername="/plots/",
    save_figure=True,
) -> None:
    """
    Visualization of the critic in a N-dimensional state space
    The N-dimensional state space is projected into its first two dimensions.
    A FeatureInverter wrapper should be used to select which features to put first so as to plot them
    :param model: the policy and critic to be plotted
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
                if hasattr(model, "critic"):
                    # For REINFORCE
                    if isinstance(model, REINFORCE):
                        value = model.critic.forward(obs_as_tensor(obs, model.device))
                    elif isinstance(model, DDPG):
                        action = model.forward(obs_as_tensor(obs, model.device))
                        value = model.critic.forward(
                            obs_as_tensor(obs, model.device), action
                        )
                    else:
                        print("visu critic: algo not covered")
                else:
                    # For A2C/PPO/DDPG
                    action = model.forward(obs_as_tensor(obs, model.device))
                    value = model.predict_values(
                        obs_as_tensor(obs, model.device), action
                    )

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
    final_show(save_figure, plot, figname, x_label, y_label, "V Function", foldername)
