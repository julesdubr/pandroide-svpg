import datetime
import os
from pathlib import Path
import torch

from omegaconf import OmegaConf

from svpg.algos import A2C, SVPG

dtime = datetime.datetime.now().strftime("/%y-%m-%d/%H-%M-%S/")
params = {
    "save_best": True,
    "logger": {
        "classname": "salina.logger.TFLogger",
        "log_dir": str(Path(__file__).parent) + "/tmp/" + dtime,
        "verbose": True,
        "cache_size": 10000,
        "every_n_seconds": 10,
    },
    "algorithm": {
        "n_particles": 16,
        "seed": 5,
        "n_envs": 1,
        "n_steps": 8,
        "n_evals": 10,
        "eval_interval": 10,
        "clipped": True,
        "gae": 0.8,
        "max_epochs": 100,
        "discount_factor": 0.95,
        "policy_coef": 1,
        "entropy_coef": 2.55e-7,
        "critic_coef": 0.4,
        "architecture": {"hidden_size": [64, 64]},
    },
    "gym_env": {
        "classname": "svpg.agents.env.make_gym_env",
        "env_name": "MyCartPoleSwingUp-v0",
        "max_episode_steps": 200,
    },
    "optimizer": {"classname": "torch.optim.RMSprop", "lr": 0.004},
}

if __name__ == "__main__":
    config = OmegaConf.create(params)

    directory = (
        str(Path(__file__).parents[1]) + "/runs/" + config.gym_env.env_name + dtime
    )

    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.manual_seed(config.algorithm.seed)

    # --------- A2C INDEPENDENT --------- #
    a2c = A2C(config)
    a2c.run(directory)

    # --------- A2C-SVPG --------- #
    svpg = SVPG(A2C(config), is_annealed=False)
    svpg.run(directory)

    # --------- A2C-SVPG_annealed --------- #
    svpg_annealed = SVPG(A2C(config), is_annealed=True)
    svpg_annealed.run(directory)
