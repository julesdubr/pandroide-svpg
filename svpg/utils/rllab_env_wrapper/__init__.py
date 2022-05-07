import gym


def register(id, entry_point, max_episode_steps=500):
    env_specs = gym.envs.registry.env_specs

    if id in env_specs.keys():
        del env_specs[id]

    gym.register(id=id, entry_point=entry_point, max_episode_steps=max_episode_steps)


register(
    id="MyCartPole-v0",
    entry_point="svpg.utils.rllab_env_wrapper.envs.rllab_cartpole:MyCartPole",
)
register(
    id="MyCartPoleSwingUp-v0",
    entry_point="svpg.utils.rllab_env_wrapper.envs.rllab_cartpoleswingup:MyCartPoleSwingUp",
)
register(
    id="MyMountainCar-v0",
    entry_point="svpg.utils.rllab_env_wrapper.envs.rllab_mountaincar:MyMountainCar",
)
register(
    id="MyPendulum-v0",
    entry_point="svpg.utils.rllab_env_wrapper.envs.rllab_pendulum:MyPendulum",
)
