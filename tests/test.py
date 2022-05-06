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


@hydra.main(config_path=".", config_name="test.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except:
        pass

    directory = str(Path(__file__).parents[1])

    if not os.path.exists(directory):
        os.makedirs(directory)

    
    algo = instantiate(cfg.algorithm)
    svpg = SVPG(algo)
    svpg.run(directory)
    # algo.run(directory)

if __name__ == "__main__":
    main()