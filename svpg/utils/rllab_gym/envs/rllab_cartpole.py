import rllab.envs.box2d.cartpole_env as rllab_cartpole
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.envs.box2d.parser import find_body
from rllab.core.serializable import Serializable
from rllab.misc import autoargs

import gym

from .get_model_path import model_path


class CartPole(rllab_cartpole.CartpoleEnv, gym.Env):
    @autoargs.inherit(Box2DEnv.__init__)
    def __init__(self, *args, **kwargs):
        self.max_pole_angle = 0.2
        self.max_cart_pos = 2.4
        self.max_cart_speed = 4.0
        self.max_pole_speed = 4.0
        self.reset_range = 0.05
        super(rllab_cartpole.CartpoleEnv, self).__init__(
            model_path("cartpole.xml.mako"), *args, **kwargs
        )
        self.cart = find_body(self.world, "cart")
        self.pole = find_body(self.world, "pole")
        Serializable.__init__(self, *args, **kwargs)
