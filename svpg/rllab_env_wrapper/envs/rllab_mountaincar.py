import rllab.envs.box2d.mountain_car_env as rllab_mountaincar
from rllab.envs.box2d.parser import find_body
from rllab.core.serializable import Serializable

import gym

from svpg.rllab_env_wrapper.envs.get_model_path import model_path

class MyMountainCar(rllab_mountaincar.MountainCarEnv, gym.Env):
    def __init__(self, 
                height_bonus=1.,
                goal_cart_pos=.6,
                *args, **kwargs):
        super(rllab_mountaincar.MountainCarEnv, self).__init__(
            model_path("mountain_car.xml.mako"),
            *args, **kwargs
        )
        self.max_cart_pos = 2
        self.goal_cart_pos = goal_cart_pos
        self.height_bonus = height_bonus
        self.cart = find_body(self.world, "cart")
        Serializable.quick_init(self, locals())