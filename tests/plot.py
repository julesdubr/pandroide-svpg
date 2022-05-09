import numpy as np
import matplotlib.pyplot as plt

with open("/home/khiemlk17/Documents/M1 S2 ANDROIDE/Project/pandroide-svpg/outputs/2022-05-09/00-53-14/svpg_normal/rewards.npy", "rb") as f:
    rewards = np.load(f)

reward = rewards[0]

plt.plot(range(reward.shape[0]), reward)
plt.show()