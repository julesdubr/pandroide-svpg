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

import pickle


@hydra.main(config_path=".", config_name="test_a2c.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except:
        pass

    d = datetime.datetime.today()
    directory = d.strftime(str(Path(__file__).parents[1]) + "/archives/%m-%d_%H-%M/")

    env = instantiate(cfg.algorithm.env)
    env_name = cfg.env_name

    # --------- A2C INDEPENDENT --------- #
    algo_a2c = instantiate(cfg.algorithm)
    algo_a2c.run()

    a2c_reward = algo_a2c.rewards
    a2c_best_rewards = [max(r) for r in a2c_reward.values()]

    plot_algo_policies(algo_a2c, env, env_name, directory + "/A2C_INDEPENDANT/")

    # ----------- SVPG NORMAL ----------- #
    algo_svpg_normal = instantiate(cfg.algorithm, clipped="False")
    svpg_normal = SVPG(algo_svpg_normal, is_annealed=False)
    svpg_normal.run()

    svpg_normal_reward = svpg_normal.algo.rewards

    plot_algo_policies(svpg_normal.algo, env, env_name, directory + "/SVPG_NORMAL/")

    # ------ SVPG CLIPPED ANNEALED ------ #
    algo_svpg_clipped_annealed = instantiate(cfg.algorithm)
    svpg_clipped_annealed = SVPG(algo_svpg_clipped_annealed)
    svpg_clipped_annealed.run()

    svpg_clipped_annealed_reward = svpg_clipped_annealed.algo.rewards

    plot_algo_policies(
        svpg_normal.algo, env, env_name, directory + "/SVPG_CLIPPED_ANNEALED/"
    )

    # ------------ HISTOGRAM ------------ #

    svpg_normal_best_rewards = [r[-1] for r in svpg_normal_reward.values()]
    svpg_annealed_best_rewards = [r[-1] for r in svpg_clipped_annealed_reward.values()]

    plot_histograms(
        [a2c_best_rewards, svpg_normal_best_rewards, svpg_annealed_best_rewards],
        [f"A2C-independent", f"A2C-SVPG", f"A2C-SVPG (clipped + annealed)"],
        ["red", "blue", "green"],
        "A2C",
        directory,
    )

    eval_interval = algo_a2c.eval_interval

    max_a2c_reward = a2c_reward[
        max(a2c_reward, key=lambda particle: sum(a2c_reward[particle]))
    ]
    max_svpg_normal_reward = svpg_normal_reward[
        max(svpg_normal_reward, key=lambda particle: sum(svpg_normal_reward[particle]))
    ]
    max_svpg_clipped_annealed_reward = svpg_clipped_annealed_reward[
        max(
            svpg_clipped_annealed_reward,
            key=lambda particle: sum(svpg_clipped_annealed_reward[particle]),
        )
    ]

    a2c_timesteps = np.array([range(len(max_a2c_reward))])
    a2c_timesteps = (a2c_timesteps + 1) * eval_interval
    a2c_timesteps = np.squeeze(a2c_timesteps, 0)

    svpg_normal_timesteps = np.array([range(len(max_svpg_normal_reward))])
    svpg_normal_timesteps = (svpg_normal_timesteps + 1) * eval_interval
    svpg_normal_timesteps = np.squeeze(svpg_normal_timesteps, 0)

    svpg_clipped_annealed_timesteps = np.array(
        [range(len(max_svpg_clipped_annealed_reward))]
    )
    svpg_clipped_annealed_timesteps = (
        svpg_clipped_annealed_timesteps + 1
    ) * eval_interval
    svpg_clipped_annealed_timesteps = np.squeeze(svpg_clipped_annealed_timesteps, 0)

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
    plt.show()


if __name__ == "__main__":
    main()
