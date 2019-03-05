from .quadcopter_env import QuadcopterGym
from gym import spaces
import logging

import numpy as np

OBSERVATION_SPACE = (16,)
ACTION_SPACE = (4,)

DIST_THRESHOLD = 0.5
TIME_THRESHOLD = 2000

LOWER_BOUNDS = np.array([-2.5, -2.5, 0.0])
UPPER_BOUNDS = np.array([2.5, 2.5, 5])

log = logging.getLogger(__name__)


class PositionControlEnv(QuadcopterGym):

    def __init__(self):
        self.observation_space = spaces.Box(-np.inf,
                                            np.inf, OBSERVATION_SPACE, dtype=np.float)
        self.action_space = spaces.Box(-np.inf,
                                       np.inf, ACTION_SPACE, dtype=np.float)

        quad = self._get_fresh_quad()
        self.quad_name = 'quad'
        super().__init__({self.quad_name: quad})

        self.goal = np.random.uniform(LOWER_BOUNDS + 0.5, UPPER_BOUNDS - 0.5)

    def _get_fresh_quad(self):
        quad = {
            'position': np.random.uniform(LOWER_BOUNDS + 0.5, UPPER_BOUNDS - 0.5).tolist(),
            'orientation': [0, 0, 0],
            'L': 0.3, 'r': 0.1, 'prop_size': [10, 4.5], 'weight': 1.2,
        }
        log.debug("Initialize new quad: {}".format(quad))
        return quad

    def _do_action(self, action: np.ndarray):
        assert action.shape == ACTION_SPACE
        action = action.tolist()

        log.debug('Setting action: {}'.format(action))

        self.quad_sim.set_motor_speeds(self.quad_name, action)

    def _get_obs(self):
        pos = self.quad_sim.get_position(self.quad_name)
        ang = self.quad_sim.get_orientation(self.quad_name)
        lv = self.quad_sim.get_linear_rate(self.quad_name)
        av = self.quad_sim.get_angular_rate(self.quad_name)

        goal = self.goal
        collision = self._get_collision()

        return np.concatenate((
            pos, ang, lv, av,
            goal, collision.astype(int)
        ), axis=None).astype(float)

    def _get_collision(self):
        pos = self.quad_sim.get_position(self.quad_name)
        return np.array([pos <= LOWER_BOUNDS, pos >= UPPER_BOUNDS]).any()

    def _get_reward(self):
        pos = self.quad_sim.get_position(self.quad_name)
        ang = self.quad_sim.get_orientation(self.quad_name)
        lv = self.quad_sim.get_linear_rate(self.quad_name)
        av = self.quad_sim.get_angular_rate(self.quad_name)

        goal = self.goal
        collision = self._get_collision()

        p_t = np.linalg.norm(pos - goal)
        angle_t = np.linalg.norm(ang)
        vel_t = np.linalg.norm(lv)
        avel_t = np.linalg.norm(av)

        cost_t = 4e-3 * p_t + 5e-4 * vel_t + 2e-4 * angle_t + 3e-4 * avel_t

        return -cost_t

    def _get_done(self):
        ob = self._get_collision()
        return (self.timestep >= TIME_THRESHOLD) or ob

    def _do_reset(self):
        # first reset quad
        quad = self._get_fresh_quad()
        self.quad_sim.set_position(self.quad_name, quad['position'])
        self.quad_sim.set_orientation(self.quad_name, quad['orientation'])

        # Then reset goal position
        self.goal = np.random.uniform(LOWER_BOUNDS + 0.5, UPPER_BOUNDS - 0.5)

    def _custom_render(self):
        if 'goal' not in self.gui.points:
            self.gui.add_point('goal')
        self.gui.points['goal']['point'] = self.goal


