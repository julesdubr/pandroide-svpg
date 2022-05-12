import gym


def register(id, entry_point, max_episode_steps=500):
    env_specs = gym.envs.registry.env_specs

    if id in env_specs.keys():
        del env_specs[id]

    gym.register(id=id, entry_point=entry_point, max_episode_steps=max_episode_steps)


register(
    id="RllCartPole-v0",
    entry_point="svpg.utils.rllab_gym.envs.rllab_cartpole:CartPole",
)
register(
    id="RllCartPoleSwingUp-v0",
    entry_point="svpg.utils.rllab_gym.envs.rllab_cartpoleswingup:CartPoleSwingUp",
)
register(
    id="RllMountainCar-v0",
    entry_point="svpg.utils.rllab_gym.envs.rllab_mountaincar:MountainCar",
)
register(
    id="RllPendulum-v0",
    entry_point="svpg.utils.rllab_gym.envs.rllab_pendulum:Pendulum",
)
