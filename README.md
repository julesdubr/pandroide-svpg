# pandroide-svpg

Stein Variational Policy Gradient (or SVPG) is a reinforcement learning (RL) method of learning and exploring multiple policies. Several agents (“workers”) work in parallel, which speeds up the exploration. The goal of SVPG is to prevent these workers from learning the same solution by distancing them from each other, allowing for a greater diversity of exploration.

Thus, the goal of our project is to modernize this algorithm using more modern tools such as PyTorch, SaLiNa and OpenIA Gym, and to compare it with others (Impala, Gorilla, etc.) in order to highlight the relevant use cases.

## Resources

- Our [colab notebook](https://colab.research.google.com/drive/15Kv6SnBmB3NXLfmZnPS88TnpEqXGDLvZ#scrollTo=SqNaC7QC_GwF).
- The [original colab](https://colab.research.google.com/drive/1foozXbDd4YNYuYKdjwFIcwiUnIaR7-Or?usp=sharing#scrollTo=SqNaC7QC_GwF) of the SVGD algorithm.
- The [PyTorch](https://pytorch.org/) library.
- The [SaLiNa](https://github.com/facebookresearch/salina) library.
- The [Gym](https://gym.openai.com/) toolkit.

Our code was inspired from [this github repo](https://github.com/largelymfs/svpg_REINFORCE). It was written by one of the authors of [this paper](https://arxiv.org/pdf/1704.02399.pdf) (Yang Liu).

## Team PANDROIDE
- CANITROT Julien
- DUBREUIL Jules
- HUYNH Tan Khiem
- KOSTADINOVIC Nikola
