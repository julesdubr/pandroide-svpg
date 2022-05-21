# pandroide-svpg

Stein Variational Policy Gradient (SVPG) (Y. Liu et al. 2017), is a reinforcement 
learning (RL) method (Sutton and Barto 2018), which enables the learning and 
exploitation of several policies. Several agents (later called "particles") work in 
parallel, which speeds up exploration. The advantage of SVPG is that it prevents 
these particles from learning the same solution by moving them away from each other, 
which promotes a greater diversity of solutions.

The goal of our project is, under the supervision of Olivier Sigaud and Oliver 
Serris, to modernize this algorithm using more modern tools such as PyTorch Paszke 
et al. 2019), SaLinA (Denoyer et al. 2021) and OpenIA Gym (Brockman et al. 2016), 
compare it to other conventional Policy Gradient methods such as Advantage Actor 
Critic (A2C) (Mnih et al. 2016) or REINFORCE (Williams 1992), develop visualization 
tools to reproduce the results obtained in the original article and highlight 
relevant use cases.


## Resources

- Our [colab notebook](https://colab.research.google.com/drive/15Kv6SnBmB3NXLfmZnPS88TnpEqXGDLvZ#scrollTo=SqNaC7QC_GwF).
- The [original colab](https://colab.research.google.com/drive/1foozXbDd4YNYuYKdjwFIcwiUnIaR7-Or?usp=sharing#scrollTo=SqNaC7QC_GwF) of the SVGD algorithm.
- The [PyTorch](https://pytorch.org/) library.
- The [SaLiNa](https://github.com/facebookresearch/salina) library.
- The [Gym](https://gym.openai.com/) toolkit.

Our code was inspired from [this github repo](https://github.com/largelymfs/svpg_REINFORCE). It was written by one of the authors of [this paper](https://arxiv.org/pdf/1704.02399.pdf) (Yang Liu).

## Installation

```
pip install -e .
```

We also recommand installing the following librairies :
[my_gym](https://github.com/osigaud/my_gym),
[rllab](https://github.com/rll/rllab),
[pybox2D](https://github.com/pybox2d/pybox2d)
and osigaud's fork of [salina](https://github.com/osigaud/salina).

## Usage

Examples are available in the `tests` directory. In practice, we use a configuration that allows us to define the hyper-parameters of the algorithms, the structure of the neural networks of the agents, the environment, optimizer...

```py
config = OmegaConf.create({
    "logger": {
        "classname": "salina.logger.TFLogger",
        "log_dir": "./tmp/",
        "verbose": False,
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
})  # It is also possible tu use Hydra
```

We can then create algorithms defined in the package `svpg.algos`. For SVPG, the `is_annealed` parameter allows you to activate or not the _annealing_ option. 
Use its `run` method to run it.

```py
from svpg.algos import A2C, SVPG

a2c = A2C(config) # A2C-Independent
a2c.run()
SVPG(A2C(config), is_annealed=False).run()  # A2C-SVPG
SVPG(A2C(config), is_annealed=True).run()  # A2C-SVPG_annealed
```

## Team PANDRO-SVPG
CANITROT Julien, DUBREUIL Jules, HUYNH Tan Khiem, KOSTADINOVIC Nikola
