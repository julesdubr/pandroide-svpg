import hydra

from svpg_a2c_mono import SVPG_A2C_Mono

@hydra.main(config_path=".", config_name="test.yaml")
def main(cfg):
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")

    algo = SVPG_A2C_Mono(cfg)
    algo.run_svpg()


if __name__ == "__main__":
    main()