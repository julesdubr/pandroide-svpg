import hydra

from svpg.algos import A2C, SVPG
from svpg.common.kernel import RBF
from svpg.common.visu import plot_histograms, plot_cartpole

from pathlib import Path


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    a2c = A2C(cfg)
    indep_rewards = a2c.run()

    svpg = SVPG(cfg, a2c, RBF)
    svpg_rewards = svpg.run()

    directory = str(Path(__file__).parents[1]) + "/plots/"
    plot_histograms(indep_rewards, svpg_rewards, "A2C", directory, plot=False)

    for pid in range(a2c.n_particles):
        plot_cartpole(
            a2c.action_agents[pid],
            a2c.env,
            f"policy_{pid}.png",
            directory,
            plot=False,
            stochastic=True,
        )
        plot_cartpole(
            a2c.critic_agents[pid],
            a2c.env,
            f"critic_{pid}.png",
            directory,
            plot=False,
        )


if __name__ == "__main__":
    main()
