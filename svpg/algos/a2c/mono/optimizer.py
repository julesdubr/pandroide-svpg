from salina import get_arguments, get_class
import torch.nn as nn


def setup_optimizers(cfg, prob_agent, critic_agent):
    """
    Setup the optimizer for salina by using the arguments in the configuration file
    """
    optimizer_args = get_arguments(cfg.algorithm.optimizer)
    parameters = nn.Sequential(prob_agent, critic_agent).parameters()
    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
    return optimizer
