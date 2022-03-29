from salina import get_arguments, get_class
import torch.nn as nn

# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, reinforce_agents):
    optimizer_args = get_arguments(cfg.algorithm.optimizer)

    optimizers = [
        get_class(cfg.algorithm.optimizer)(
            reinforce_agent.parameters(), **optimizer_args
        )
        for reinforce_agent in reinforce_agents
    ]

    return optimizers
