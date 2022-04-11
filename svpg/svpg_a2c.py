import hydra

from torch import autograd

from svpg.algos import A2C, SVPG
from svpg.common.kernel import RBF
from svpg.common.visu import plot_histograms, plot_cartpole

from pathlib import Path
import datetime


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg):
    import torch.multiprocessing as mp

    mp.set_start_method("spawn")

    a2c = A2C(cfg)
    indep_rewards = a2c.run()

    svpg = SVPG(cfg, a2c, RBF)
    svpg_rewards = svpg.run()

    d = datetime.datetime.today()
    directory = d.strftime(str(Path(__file__).parents[1]) + "/archives/%m-%d_%H-%M/")
    plot_histograms(indep_rewards, svpg_rewards, "A2C", directory, plot=False)
    env = svpg.env

    for pid in range(a2c.n_particles):
        action_model = svpg.action_agents[pid].model
        figname = f"policy_{pid}.png"
        plot_cartpole(
            action_model, env, figname, directory, plot=False, stochastic=True
        )

        critic_model = svpg.critic_agents[pid].model
        figname = f"critic_{pid}.png"
        plot_cartpole(critic_model, env, figname, directory, plot=False)


if __name__ == "__main__":
    main()
