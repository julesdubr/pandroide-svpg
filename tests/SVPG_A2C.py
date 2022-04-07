import hydra

from svpg.algos.svpg import SVPG
from svpg.algos.a2c import A2C
from svpg.visu.visu_critic import plot_cartpole_critic
from svpg.visu.visu_policies import plot_histograms


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    a2c = A2C(cfg)
    svpg = SVPG(cfg, a2c)

    indep_rewards = a2c.run()
    svpg_rewards = svpg.run()

    plot_histograms(indep_rewards, svpg_rewards, "A2C")
    # plot_cartpole_critic(algo.critic_agents[0].model, algo.env_agents[0].env)


if __name__ == "__main__":
    main()
