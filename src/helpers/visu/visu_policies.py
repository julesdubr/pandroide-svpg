import os
import random

import matplotlib.pyplot as plt
import numpy as np


def final_show(
    save_figure, plot, figure_name, x_label, y_label, title, directory
) -> None:
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
        directory = os.getcwd() + "/data" + directory
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(directory + figure_name)
    if plot:
        plt.show()
    plt.close()
