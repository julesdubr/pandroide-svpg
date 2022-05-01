import rllab.envs.box2d.cartpole_env as rllab_cartpole
from rllab.envs.box2d.parser import find_body
from rllab.core.serializable import Serializable

import gym

from rllab_env_wrapper.envs.get_model_path import model_path

class MyCartPole(rllab_cartpole.CartpoleEnv, gym.Env):
    def __init__(self, *args, **kwargs):
        self.max_pole_angle = .2
        self.max_cart_pos = 2.4
        self.max_cart_speed = 4.
        self.max_pole_speed = 4.
        self.reset_range = 0.05
        super(rllab_cartpole.CartpoleEnv, self).__init__(
            model_path("cartpole.xml.mako"),
            *args, **kwargs
        )
        self.cart = find_body(self.world, "cart")
        self.pole = find_body(self.world, "pole")
        Serializable.__init__(self, *args, **kwargs)