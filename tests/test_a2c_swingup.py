import datetime
import os
from pathlib import Path
import torch

from omegaconf import OmegaConf

from svpg.algos import A2C, SVPG

dtime = datetime.datetime.now().strftime("/%y-%m-%d/%H-%M-%S/")
params = {
    "save_run": True,
    "logger": {
        "classname": "salina.logger.TFLogger",
        "log_dir": str(Path(__file__).parent) + "/tmp/" + dtime,
        "verbose": False,
        "cache_size": 10000,
        "every_n_seconds": 10,
    },
    "algorithm": {
        "n_particles": 16,
        "seed": 432,
        "n_envs": 8,
        "n_steps": 16,
        "eval_interval": 80,
        "n_evals": 10,
        "clipped": True,
        "max_epochs": 16000,
        "discount_factor": 0.99,
        "gae_coef": 1,
        "policy_coef": 1.0,
        "critic_coef": 1.0,
        "entropy_coef": 1e-5,
        "architecture": {"hidden_size": [100, 50, 25]},
    },
    "gym_env": {
        "classname": "svpg.agents.env.make_gym_env",
        "env_name": "RllCartPoleSwingUp-v0",
    },
    "optimizer": {"classname": "torch.optim.Adam", "lr": 5e-3},
}

if __name__ == "__main__":
    config = OmegaConf.create(params)

    directory = (
        str(Path(__file__).parents[1]) + "/runs/" + config.gym_env.env_name + dtime
    )

    if not os.path.exists(directory):
        os.makedirs(directory)

    # torch.manual_seed(config.algorithm.seed)

    # --------- A2C INDEPENDENT --------- #
    a2c = A2C(config)
    a2c.run(directory)

    # --------- A2C-SVPG --------- #
    svpg = SVPG(A2C(config), is_annealed=False)
    svpg.run(directory)

    # --------- A2C-SVPG_annealed --------- #
    svpg_annealed = SVPG(A2C(config), is_annealed=True)
    svpg_annealed.run(directory)
