import datetime
import time
import os
from pathlib import Path
import torch

from omegaconf import OmegaConf

from svpg.algos import A2C

params = {
    "logger": {
        "classname": "salina.logger.TFLogger",
        "log_dir": "./tmp/" + str(time.time()),
        "verbose": True,
        # "cache_size": 10000,
        "every_n_seconds": 10,
    },
    "algorithm": {
        "n_particles": 16,
        "seed": 4,
        "n_envs": 1,
        "n_steps": 20,
        "n_evals": 1,
        "eval_interval": 1,
        "clipped": True,
        "gae": 0.8,
        "max_epochs": 10,
        "discount_factor": 0.95,
        "policy_coef": 0.1,
        "entropy_coef": 0.001,
        "critic_coef": 1.0,
        "architecture": {"hidden_size": [25, 25]},
    },
    "gym_env": {
        "classname": "svpg.agents.env.make_gym_env",
        "env_name": "CartPole-v1",
        "max_episode_steps": 500,
    },
    "optimizer": {"classname": "torch.optim.Adam", "lr": 0.01},
}

if __name__ == "__main__":
    d = datetime.datetime.now()
    directory = d.strftime(
        str(Path(__file__).parents[1]) + "/archives/%y-%m-%d/%H-%M-%S/"
    )

    if not os.path.exists(directory):
        os.makedirs(directory)

    config = OmegaConf.create(params)
    torch.manual_seed(config.algorithm.seed)

    # --------- A2C INDEPENDENT --------- #
    a2c = A2C(config)
    a2c.run(directory)
