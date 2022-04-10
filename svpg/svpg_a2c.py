import hydra

from svpg.algos import A2C, SVPG
from svpg.common.kernel import RBF
from svpg.common.visu import plot_histograms, plot_cartpole


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    a2c = A2C(cfg)
    # indep_rewards = a2c.run()

    svpg = SVPG(cfg, a2c, RBF)
    svpg_rewards = svpg.run()

    # plot_histograms(indep_rewards, svpg_rewards, "A2C")
    # plot_cartpole(algo.critic_agents[0], algo.env_agents[0].env)


if __name__ == "__main__":
    main()
