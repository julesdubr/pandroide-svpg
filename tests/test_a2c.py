import hydra
from hydra.utils import instantiate

import torch

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
        torch.manual_seed(cfg.algorithm.env_seed)
    except:
        pass

    directory = directory = str(Path(__file__).parents[1])

    if not os.path.exists(directory):
        os.makedirs(directory)

    # --------- A2C INDEPENDENT --------- #
    algo_a2c = instantiate(cfg.algorithm)
    algo_a2c.run(directory)
    
    # --------- SVPG Normal ------------- #
    algo_svpg_normal = instantiate(cfg.algorithm)
    svpg_normal = SVPG(algo_svpg_normal, is_annealed=False)
    svpg_normal.run(directory)
    
    # --------- SVPG Annealed ------------- #
    algo_svpg_annealed = instantiate(cfg.algorithm)
    svpg_annealed = SVPG(algo_svpg_annealed)
    svpg_annealed.run(directory)


if __name__ == "__main__":
    main()
