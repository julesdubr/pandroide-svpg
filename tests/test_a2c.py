from cProfile import label
import hydra
from hydra.utils import instantiate

from svpg.algos.svpg import SVPG
from svpg.common.visu import plot_histograms

import matplotlib.pyplot as plt
import numpy as np

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)


@hydra.main(config_path=".", config_name="test_a2c.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    algo = instantiate(cfg.algorithm)
    algo.run()

    # algo_svpg = instantiate(cfg.algorithm)
    # svpg = SVPG(algo_svpg)
    # svpg.run()
    
    # plot_histograms(algo.rewards[-1], svpg.algo.rewards[-1], title="Cartpole-v0")

    max_reward_a2c = max(algo.rewards.T.tolist(), key=lambda x: sum(x))
    # max_reward_svpg = max(svpg.algo.rewards.T.tolist(), key=lambda x: sum(x))

    max_epochs = algo.max_epochs

    plt.figure()
    plt.plot(range(max_epochs), max_reward_a2c, label="A2C independent")
    # plt.plot(range(max_epochs), max_reward_svpg, label="A2C-SVPG")
    plt.title("CartpoleContinuous-v1")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Average cummulated reward")
    plt.show()
    plt.savefig("../plots/cartpole-v0")


if __name__ == "__main__":
    # with autograd.detect_anomaly():
    #     main()
    main()
