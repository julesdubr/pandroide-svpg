from salina import get_arguments, get_class

# Configure the optimizer over the a2c agent
def setup_optimizers(cfg, prob_agents, critic_agents):
    optimizer_args = get_arguments(cfg.algorithm.optimizer)

    parameters = []
    for prob_agent, critic_agent in zip(prob_agents, critic_agents):
        parameters += list(prob_agent.parameters()) + list(critic_agent.parameters())

    optimizer = get_class(cfg.algorithm.optimizer)(parameters, **optimizer_args)
    return optimizer