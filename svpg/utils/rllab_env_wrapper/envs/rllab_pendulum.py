import rllab.envs.box2d.double_pendulum_env as rllab_pendulum
from rllab.envs.box2d.box2d_env import Box2DEnv
from rllab.envs.box2d.parser import find_body
from rllab.core.serializable import Serializable
from rllab.misc import autoargs

import gym

import numpy as np

from .get_model_path import model_path


class MyPendulum(rllab_pendulum.DoublePendulumEnv, gym.Env):
    @autoargs.inherit(Box2DEnv.__init__)
    def __init__(self, *args, **kwargs):
        kwargs["frame_skip"] = kwargs.get("frame_skip", 2)
        if kwargs.get("template_args", {}).get("noise", False):
            self.link_len = (np.random.rand() - 0.5) + 1
        else:
            self.link_len = 1
        kwargs["template_args"] = kwargs.get("template_args", {})
        kwargs["template_args"]["link_len"] = self.link_len
        super(rllab_pendulum.DoublePendulumEnv, self).__init__(
            model_path("double_pendulum.xml.mako"), *args, **kwargs
        )
        self.link1 = find_body(self.world, "link1")
        self.link2 = find_body(self.world, "link2")
        Serializable.__init__(self, *args, **kwargs)
