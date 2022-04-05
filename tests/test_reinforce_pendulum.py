import hydra

from svpg.algos.svpg_reinforce_mono import SVPG_Reinforce_Mono
from svpg.visu.visu_critic import plot_pendulum_critic


@hydra.main(config_path=".", config_name="test_reinforce_pendulum.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    algo = SVPG_Reinforce_Mono(cfg)
    algo.run_svpg()

    plot_pendulum_critic(algo.critic_agents[0].model, algo.env_agents[0].env)


if __name__ == "__main__":
    main()
