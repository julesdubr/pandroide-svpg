import hydra
from hydra.utils import instantiate

from svpg.algos.svpg import SVPG

from omegaconf import OmegaConf

import matplotlib.pyplot as plt

import datetime
from pathlib import Path

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

    directory = directory = str(Path(__file__).parents[1])

    if not os.path.exists(directory):
        os.makedirs(directory)

    # --------- A2C INDEPENDENT --------- #
    algo_a2c = instantiate(cfg.algorithm)
    algo_a2c.run(directory)

    a2c_rewards = algo_a2c.rewards
    # ------------------------------------ #

    # ----------- SVPG NORMAL ----------- #
    algo_svpg_normal = instantiate(cfg.algorithm, clipped="False")
    svpg_normal = SVPG(algo_svpg_normal, is_annealed=False)
    svpg_normal.run(directory)

    svpg_normal_rewards = svpg_normal.algo.rewards
    # ------------------------------------ #

    # ------ SVPG CLIPPED ANNEALED ------ #
    algo_svpg_clipped_annealed = instantiate(cfg.algorithm)
    svpg_clipped_annealed = SVPG(algo_svpg_clipped_annealed)
    svpg_clipped_annealed.run(directory)

    svpg_clipped_annealed_rewards = svpg_clipped_annealed.algo.rewards
    # ------------------------------------ #

    # ------------ Compare best agents ------------ #
    max_a2c_reward_index = max(a2c_rewards, 
                               key=lambda particle: sum(a2c_rewards[particle]))

    max_a2c_reward = a2c_rewards[max_a2c_reward_index]

    max_svpg_normal_reward_index = max(svpg_normal_rewards, 
                                       key=lambda particle: sum(svpg_normal_rewards[particle]))

    max_svpg_normal_reward = svpg_normal_rewards[max_svpg_normal_reward_index]

    max_svpg_clipped_annealed_reward_index = max(svpg_clipped_annealed_rewards,
                                           key=lambda particle: sum(svpg_clipped_annealed_rewards[particle]),
                                          )
    
    max_svpg_clipped_annealed_reward = svpg_clipped_annealed_rewards[max_svpg_clipped_annealed_reward_index]

    a2c_timesteps = algo_a2c.eval_epoch[max_a2c_reward_index]

    svpg_normal_timesteps = svpg_normal.algo.eval_epoch[max_svpg_normal_reward_index]

    svpg_clipped_annealed_timesteps = svpg_clipped_annealed.algo.eval_epoch[max_svpg_clipped_annealed_reward_index]

    plt.figure()
    plt.plot(a2c_timesteps, max_a2c_reward, label="A2C")
    plt.plot(svpg_normal_timesteps, max_svpg_normal_reward, label="SVPG_A2C_normal")
    plt.plot(
        svpg_clipped_annealed_timesteps,
        max_svpg_clipped_annealed_reward,
        label="SVPG_A2C_clipped_annealed",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.legend()
    plt.title(cfg.algorithm.env_name)
    plt.savefig(directory + "A2C_SVPG_loss.png")
    #-----------------------------a2c


if __name__ == "__main__":
    main()
