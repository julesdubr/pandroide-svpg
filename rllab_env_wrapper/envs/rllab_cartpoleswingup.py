import rllab.envs.box2d.cartpole_swingup_env as rllab_cartpole_swingup
from rllab.envs.box2d.parser import find_body
from rllab.core.serializable import Serializable

import gym

from rllab_env_wrapper.envs.get_model_path import model_path

class MyCartPoleSwingUp(rllab_cartpole_swingup.CartpoleSwingupEnv, gym.Env):
    def __init__(self, *args, **kwargs):
        super(rllab_cartpole_swingup.CartpoleSwingupEnv, self).__init__(
            model_path("cartpole.xml.mako"),
            *args, **kwargs
        )

        self.max_cart_pos = 3
        self.max_reward_cart_pos = 3
        self.cart = find_body(self.world, "cart")
        self.pole = find_body(self.world, "pole")
        Serializable.__init__(self, *args, **kwargs)