import numpy as np
import matplotlib.pyplot as plt

with open("/home/khiemlk17/Documents/M1 S2 ANDROIDE/Project/pandroide-svpg/outputs/2022-05-12/21-54-15/algo_base/rewards.npy", "rb") as f:
    a2c_rewards = np.load(f)

with open("/home/khiemlk17/Documents/M1 S2 ANDROIDE/Project/pandroide-svpg/outputs/2022-05-12/21-54-15/svpg_normal/rewards.npy", "rb") as f:
    svpg_rewards = np.load(f)

with open("/home/khiemlk17/Documents/M1 S2 ANDROIDE/Project/pandroide-svpg/outputs/2022-05-12/21-54-15/svpg_annealed/rewards.npy", "rb") as f:
    svpg_annealed_rewards = np.load(f)

# reward = rewards[rewards.sum(1).argmax()]
# reward = rewards.sum(axis=0) / 16
a2c_reward = a2c_rewards[0]
svpg_reward = svpg_rewards[0]
svpg_annealed_reward = svpg_annealed_rewards[0]

plt.plot(range(a2c_reward.shape[0]), a2c_reward, label="A2C")
plt.plot(range(svpg_reward.shape[0]), svpg_reward, label="SVPG")
plt.plot(range(svpg_annealed_reward.shape[0]), svpg_annealed_reward, label="SVPG annealed")
plt.legend()
plt.show()