import matplotlib.pyplot as plt
from visu.visu_policies import final_show


def visu_cartpole_replay_data(list_states, list_targets) -> None:
    """
    visualize, for a list of states plotted for their first dimension,
    the corresponding target value for the critic as computed either with
    a TD or a MC method.
    In the MC case, it gives the value V(s) of being in that state
    In the TD case, the target is given by the local temporal difference error
    the state is assumed 4-dimensional (cartpole case)
    :param list_states: a list of states, usually taken from a batch
    :param list_targets: a list of target values, usually computed from a batch
    :return: nothing
    """
    states = list_states[:, (0, 2)].cpu().numpy()
    values = list_targets.cpu().numpy()

    plt.figure(figsize=(10, 4))

    scatter_plot = plt.scatter(states[:, 0], states[:, 1], c=values, s=35)
    plt.colorbar(scatter_plot)
    final_show(
        save_figure=True,
        plot=False,
        figure_name="target_landscape",
        x_label="pos",
        y_label="angle",
        title="Target Landscape",
        directory="plots",
    )


def visu_loss_along_time(cpts, losses, loss_file_name) -> None:
    """
    Plots the evolution of the loss along time
    :param cpts: step counter
    :param losses: the successive values of the loss
    :param loss_file_name: the file where to store the results
    :return: nothing
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    plt.cla()
    # ax.set_ylim(-1.0, 500.0)  # data dependent
    ax.set_title("Loss Analysis", fontsize=35)
    ax.set_xlabel("cpt", fontsize=24)
    ax.set_ylabel("loss", fontsize=24)
    ax.scatter(cpts, losses, color="blue", alpha=0.2)
    plt.savefig("./results/" + loss_file_name + ".pdf")
    plt.show()
