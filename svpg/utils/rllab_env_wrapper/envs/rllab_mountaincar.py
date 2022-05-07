import rllab.envs.box2d.mountain_car_env as rllab_mountaincar
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.envs.box2d.parser import find_body
from rllab.core.serializable import Serializable
from rllab.misc import autoargs

import gym

from .get_model_path import model_path


class MyMountainCar(rllab_mountaincar.MountainCarEnv, gym.Env):
    @autoargs.inherit(Box2DEnv.__init__)
    @autoargs.arg(
        "height_bonus_coeff",
        type=float,
        help="Height bonus added to each step's reward",
    )
    @autoargs.arg("goal_cart_pos", type=float, help="Goal horizontal position")
    def __init__(self, height_bonus=1.0, goal_cart_pos=0.6, *args, **kwargs):
        super(rllab_mountaincar.MountainCarEnv, self).__init__(
            model_path("mountain_car.xml.mako"), *args, **kwargs
        )
        self.max_cart_pos = 2
        self.goal_cart_pos = goal_cart_pos
        self.height_bonus = height_bonus
        self.cart = find_body(self.world, "cart")
        Serializable.quick_init(self, locals())
