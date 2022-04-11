import hydra
from hydra.utils import instantiate

from svpg.algos.reinforce import REINFORCE

from omegaconf import OmegaConf
OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)


@hydra.main(config_path=".", config_name="test_reinforce.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    algo = instantiate(cfg.algorithm)
    algo.run()


if __name__ == "__main__":
    main()
