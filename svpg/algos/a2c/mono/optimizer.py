from salina import get_arguments, get_class
import torch.nn as nn

# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, prob_agents, critic_agents):
    optimizer_args = get_arguments(cfg.algorithm.optimizer)

    optimizers = list()

    for prob_agent, critic_agent in zip(prob_agents, critic_agents):
        parameters = nn.Sequential(prob_agent, critic_agent).parameters()
        optimizers.append(
            get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
        )

    return optimizers
