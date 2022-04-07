import hydra

from svpg.algos.svpg_a2c_mono import SVPG_A2C_Mono
from svpg.common.visu import plot_histograms, plot_cartpole


@hydra.main(config_path=".", config_name="test_a2c.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    algo = SVPG_A2C_Mono(cfg)

    algo.run()
    indep_rewards = algo.rewards

    algo = SVPG_A2C_Mono(cfg)

    algo.run_svpg()
    svpg_rewards = algo.rewards

    plot_histograms(indep_rewards, svpg_rewards, "A2C")


if __name__ == "__main__":
    main()
