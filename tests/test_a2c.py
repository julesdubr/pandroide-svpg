import hydra
from hydra.utils import instantiate

from svpg.algos.svpg import SVPG

from omegaconf import OmegaConf

import matplotlib.pyplot as plt
import numpy as np
import datetime
from pathlib import Path

from svpg.common.visu import plot_algo_policies, plot_histograms

try:
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
except:
    print("Already register")

import os


@hydra.main(config_path=".", config_name="test_a2c.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except:
        pass

    d = datetime.datetime.today()
    directory = d.strftime(str(Path(__file__).parents[1]) + "/archives/%m-%d_%H-%M/")


    if not os.path.exists(directory):
        os.makedirs(directory)

    env = instantiate(cfg.algorithm.env)
    env_name = cfg.env_name

    # --------- A2C INDEPENDENT --------- #
    algo_a2c = instantiate(cfg.algorithm)
    algo_a2c.to_gpu()
    algo_a2c.run()

    a2c_reward = algo_a2c.rewards
    a2c_best_rewards = [max(r) for r in a2c_reward.values()]

    plot_algo_policies(algo_a2c, env, env_name, directory + "/A2C_INDEPENDANT/")
    #------------------------------------ #

    # ----------- SVPG NORMAL ----------- #
    algo_svpg_normal = instantiate(cfg.algorithm, clipped="False")
    svpg_normal = SVPG(algo_svpg_normal, is_annealed=False)
    svpg_normal.algo.to_gpu()
    svpg_normal.run()

    svpg_normal_reward = svpg_normal.algo.rewards
    svpg_normal_best_rewards = [max(r) for r in svpg_normal_reward.values()]

    plot_algo_policies(svpg_normal.algo, env, env_name, directory + "/SVPG_NORMAL/")
    #------------------------------------ #

    # ------ SVPG CLIPPED ANNEALED ------ #
    algo_svpg_clipped_annealed = instantiate(cfg.algorithm)
    svpg_clipped_annealed = SVPG(algo_svpg_clipped_annealed)
    svpg_clipped_annealed.algo.to_gpu()
    svpg_clipped_annealed.run()

    svpg_clipped_annealed_reward = svpg_clipped_annealed.algo.rewards
    svpg_annealed_best_rewards = [max(r) for r in svpg_clipped_annealed_reward.values()]

    plot_algo_policies(
        svpg_normal.algo, env, env_name, directory + "/SVPG_CLIPPED_ANNEALED/"
    )
    #------------------------------------ #

    # ------------ HISTOGRAM ------------ #
    plot_histograms(
        [a2c_best_rewards, svpg_normal_best_rewards, svpg_annealed_best_rewards],
        [f"A2C-independent", f"A2C-SVPG", f"A2C-SVPG (clipped + annealed)"],
        ["red", "blue", "green"],
        "A2C",
        directory
    )
    #------------------------------------ #


    # ------------ Compare best agents ------------ #
    max_a2c_reward_index = max(a2c_reward, 
                               key=lambda particle: sum(a2c_reward[particle]))

    max_a2c_reward = a2c_reward[max_a2c_reward_index]

    max_svpg_normal_reward_index = max(svpg_normal_reward, 
                                       key=lambda particle: sum(svpg_normal_reward[particle]))

    max_svpg_normal_reward = svpg_normal_reward[max_svpg_normal_reward_index]

    max_svpg_clipped_annealed_reward_index = max(svpg_clipped_annealed_reward,
                                           key=lambda particle: sum(svpg_clipped_annealed_reward[particle]),
                                          )
    
    max_svpg_clipped_annealed_reward = svpg_clipped_annealed_reward[max_svpg_clipped_annealed_reward_index]

    a2c_timesteps = algo_a2c.eval_time_steps[max_a2c_reward_index]

    svpg_normal_timesteps = svpg_normal.algo.eval_time_steps[max_svpg_normal_reward_index]

    svpg_clipped_annealed_timesteps = svpg_clipped_annealed.algo.eval_time_steps[max_svpg_clipped_annealed_reward_index]

    plt.figure()
    plt.plot(a2c_timesteps, max_a2c_reward, label="A2C")
    plt.plot(svpg_normal_timesteps, max_svpg_normal_reward, label="SVPG_A2C")
    plt.plot(
        svpg_clipped_annealed_timesteps,
        max_svpg_clipped_annealed_reward,
        label="SVPG_A2C_clipped_annealed",
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.title(cfg.algorithm.env_name)
    plt.savefig(directory + "A2C_SVPG_loss.png")
    # plt.show()
    #------------------------------------ #


if __name__ == "__main__":
    main()
