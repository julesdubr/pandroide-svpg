import hydra
from hydra.utils import instantiate

from svpg.algos.svpg import SVPG

from omegaconf import OmegaConf

try:
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
except:
    print("Already register")


@hydra.main(config_path=".", config_name="test_rllab_env.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except:
        pass

    a2c = instantiate(cfg.algorithm)
    a2c.run()

if __name__ == "__main__":
    main()