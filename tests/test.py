import gym

import svpg.rllab_env_wrapper

env = gym.make("MyCartPoleSwingUp-v0")
print(env.reset())