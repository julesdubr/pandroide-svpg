import hydra
from omegaconf import DictConfig

from salina import instantiate_class

@hydra.main(config_path=".", config_name="test.yaml")
def main(cfg:DictConfig):
    algo = instantiate_class(cfg.algorithm)
    algo.execute_acquisition_agent(1)
    algo.execute_critic_agent()
    algo.get_policy_parameters()


if __name__ == "__main__":
    main()