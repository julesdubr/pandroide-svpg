import hydra

from algo import Algo

@hydra.main(config_path=".", config_name="test.yaml")
def main(cfg):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    algo = Algo(cfg)
    algo.execute_acquisition_agent(1)
    algo.execute_critic_agent()
    algo.get_policy_parameters()


if __name__ == "__main__":
    main()