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

    for pid in range(svpg.n_particles):
        # plot_cartpole(svpg.action_agents[pid], svpg.env, figname=f"policy_{pid}.png")
        plot_cartpole(
            svpg.critic_agents[pid],
            svpg.env,
            figname=f"critic_{pid}.png",
            directory=directory,
            plot=False,
        )


if __name__ == "__main__":
    main()
