import datetime
import os
from pathlib import Path
import torch
import pickle

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
        "seed": 4,
        "n_envs": 8,
        "n_steps": 100,
        "eval_interval": 4,
        "n_evals": 1,
        "clipped": True,
        "max_epochs": 625,
        "discount_factor": 0.95,
        "gae_coef": 0.8,
        "policy_coef": 0.1,
        "entropy_coef": 0.001,
        "critic_coef": 1.0,
        "architecture": {"hidden_size": [64, 64]},
    },
    "gym_env": {
        "classname": "svpg.agents.env.make_gym_env",
        "env_name": "CartPoleContinuous-v1",
    },
    "optimizer": {"classname": "torch.optim.Adam", "lr": 0.01},
}

if __name__ == "__main__":
    config = OmegaConf.create(params)

    directory = (
        str(Path(__file__).parents[1]) + "/runs/" + config.gym_env.env_name + dtime
    )

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + "/params.pk", "wb+") as f:
        pickle.dump(params, f)

    # torch.manual_seed(config.algorithm.seed)

    # ---------- A2C INDEPENDENT --------- #
    A2C(config).run(directory)
    # ------------- A2C-SVPG ------------- #
    SVPG(A2C(config), is_annealed=False).run(directory)
    # --------- A2C-SVPG_annealed -------- #
    SVPG(A2C(config), is_annealed=True).run(directory)
