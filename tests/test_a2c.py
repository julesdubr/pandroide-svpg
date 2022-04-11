import hydra
from hydra.utils import instantiate

from svpg.algos.svpg import SVPG

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)

from torch import autograd

@hydra.main(config_path=".", config_name="test_a2c.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    algo = instantiate(cfg.algorithm)

    svpg = SVPG(algo)
    svpg.run()


if __name__ == "__main__":
    # with autograd.detect_anomaly():
    #     main()
    main()
