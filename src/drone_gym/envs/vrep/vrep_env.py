"""Generic Gym wrappers for V-REP environments"""

from abc import ABC, abstractmethod
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from drone_gym import vrep
from drone_gym.vrep.utils import GUIItems

import logging

log = logging.getLogger(__name__)


CUR_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.abspath(os.path.join(CUR_DIR, '../../assets/vrep'))


def get_scene(scene_name):
    return os.path.join(ASSETS_DIR, 'scenes', scene_name + '.ttt')


class VREPEnv(gym.Env, ABC):
    metadata = {
        'render.modes': ['human', 'headless']
    }

    def __init__(self, *args, **kwargs):
        kwargs['quit_on_complete'] = True
        kwargs['headless'] = True
        gui_arg = os.environ.get('VREP_GUI')
        kwargs['headless'] = gui_arg not in ['YES', 'Yes', 'Y', 'ON', 'On', 'on', '1']
        log.info('VREP_GUI = {}'.format(gui_arg))
        self.sim = vrep.VREPSim(*args, **kwargs)
        self.sim.start()

        self.headless = kwargs.get('headless', False)

    def step(self, action):
        if not self.sim.started:
            raise RuntimeError('Calling step before reset?')
        self._do_action(action)
        self.sim.step_blocking_simulation()
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
        log.debug('Resetting VREPEnv')
        if not self.sim.started:
            self.sim.start()
        if not self.sim.sim_running:
            self.sim.start_blocking_simulation()
        self._do_reset()
        return self._get_obs()

    @abstractmethod
    def _do_reset(self):
        pass

    def render(self, mode='human'):
        if self.sim.sim_running and self.headless:
            log.error('Sim already running in headless mode!')
            return
        return

    def close(self):
        self.sim.stop_simulation()
        # self.sim.end()
        super().close()




