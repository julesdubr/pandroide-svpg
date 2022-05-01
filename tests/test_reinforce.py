import hydra
from hydra.utils import instantiate

from svpg.algos.svpg import SVPG

import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
except:
    print("Already register")


@hydra.main(config_path=".", config_name="test_reinforce.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except:
        pass

    algo_reinfoce = instantiate(cfg.algorithm)
    algo_reinfoce.run()

    a2c_reward = algo_reinfoce.rewards

    algo_svpg_normal = instantiate(cfg.algorithm, clipped="False")
    svpg_normal = SVPG(algo_svpg_normal, is_annealed=False)
    svpg_normal.run()

    svpg_normal_reward = svpg_normal.algo.rewards

    algo_svpg_clipped_annealed = instantiate(cfg.algorithm)
    svpg_clipped_annealed = SVPG(algo_svpg_clipped_annealed)
    svpg_clipped_annealed.run()

    svpg_clipped_annealed_reward = svpg_clipped_annealed.algo.rewards

    max_a2c_reward_index = max(a2c_reward, 
                               key=lambda particle: sum(a2c_reward[particle]))

    max_a2c_reward = a2c_reward[max_a2c_reward]

    max_svpg_normal_reward_index = max(svpg_normal_reward, 
                                       key=lambda particle: sum(svpg_normal_reward[particle]))

    max_svpg_normal_reward = svpg_normal_reward[max_svpg_normal_reward_index]

    max_svpg_clipped_annealed_reward_index = max(svpg_clipped_annealed_reward,
                                           key=lambda particle: sum(svpg_clipped_annealed_reward[particle]),
                                          )
    
    max_svpg_clipped_annealed_reward = svpg_clipped_annealed_reward[max_svpg_clipped_annealed_reward_index]

    a2c_timesteps = algo_reinfoce.eval_time_steps[max_a2c_reward_index]

    svpg_normal_timesteps = svpg_normal.algo.eval_time_steps[max_svpg_normal_reward_index]

    svpg_clipped_annealed_timesteps = svpg_clipped_annealed.algo.eval_time_steps[max_svpg_clipped_annealed_reward_index]

    plt.figure()
    plt.plot(a2c_timesteps, max_a2c_reward, label="REINFORCE")
    plt.plot(svpg_normal_timesteps, max_svpg_normal_reward, label="SVPG_REINFORCE")
    plt.plot(
        svpg_clipped_annealed_timesteps,
        max_svpg_clipped_annealed_reward,
        label="SVPG_REINFORCE_clipped_annealed",
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.title(cfg.algorithm.env_name)
    plt.show()


if __name__ == "__main__":
    main()
