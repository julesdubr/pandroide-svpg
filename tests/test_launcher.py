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


@hydra.main(config_path=".", config_name="test_launcher.yaml")
def main(cfg):
    print("In main")
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except:
        pass

    algo_reinforce = instantiate(cfg.algorithm)
    print(cfg.env_name)

if __name__ == "__main__":
    main()