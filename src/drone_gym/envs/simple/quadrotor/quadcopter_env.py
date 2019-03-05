import argparse
import logging
from abc import ABC, abstractmethod

import gym

from .gui import GUI
from .quadcopter import Quadcopter

log = logging.getLogger(__name__)

TIME_SCALING = 1.0
QUAD_DYNAMICS_UPDATE = 0.002  # 500 Hz
CONTROLLER_DYNAMICS_UPDATE = 0.005  # 200 Hz


class QuadcopterGym(gym.Env, ABC):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, quad_list: dict):
        self.quad_list = quad_list
        self.quad_sim = Quadcopter(self.quad_list)  # type: Quadcopter
        self.gui = None  # type: GUI
        self.timestep = 0

    def step(self, action):
        if self.quad_sim is None:
            raise RuntimeError('Calling step before reset?')
        self._do_action(action)
        self.quad_sim.update(CONTROLLER_DYNAMICS_UPDATE)
        self.timestep += 1

        # TODO: Empty info fields
        obs, rew, done, info = self._get_obs(), self._get_reward(), self._get_done(), dict()
        if done:
            self.reset()
        return obs, rew, done, info

    @abstractmethod
    def _do_action(self, action):
        pass

    @abstractmethod
    def _get_obs(self):
        pass

    @abstractmethod
    def _get_reward(self):
        pass

    @abstractmethod
    def _get_done(self):
        pass

    def reset(self):
        log.debug('Resetting QuadSim')
        self._do_reset()
        self.timestep = 0
        return self._get_obs()

    @abstractmethod
    def _do_reset(self):
        pass

    def render(self, mode='human'):
        if self.gui is None:
            self.gui = GUI(self.quad_list)
        for quad in self.quad_list.keys():
            self.gui.quads[quad]['position'] = self.quad_sim.get_position(quad)
            self.gui.quads[quad]['orientation'] = self.quad_sim.get_orientation(quad)
        self._custom_render()
        return self.gui.update()

    def _custom_render(self):
        pass
