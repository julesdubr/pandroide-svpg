import numpy as np
import matplotlib.pyplot as plt

with open("/home/khiemlk17/Documents/M1 S2 ANDROIDE/Project/pandroide-svpg/outputs/2022-05-09/01-26-58/algo_base/rewards.npy", "rb") as f:
    rewards = np.load(f)

reward = rewards[rewards.sum(1).argmax()]

plt.plot(range(reward.shape[0]), reward)
plt.show()