import hydra

from svpg_reinforce_mono import SVPG_Reinforce_Mono

@hydra.main(config_path=".", config_name="test_reinforce.yaml")
def main(cfg):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    algo = SVPG_Reinforce_Mono(cfg)
    algo.run_svpg()


if __name__ == "__main__":
    main()