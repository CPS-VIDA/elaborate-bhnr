import temporal_logic.signal_tl as stl

import pandas as pd

import numpy as np

import logging
from collections import deque
from typing import Callable

log = logging.getLogger(__name__)


class STLMonitor:

    def __init__(self, phi: stl.Expression, signals, rho: Callable, psiglen, dt=1):
        self.phi = phi
        self.signals = signals
        self.rho = rho
        self.psiglen = psiglen
        self.dt = dt

    def get(self, states: np.ndarray, dones: np.ndarray, n_cpu: int) -> np.ndarray:
        rewards = np.zeros_like(dones)
        for i in range(n_cpu):
            ri = self._get_one(states[:, i, :], dones[:, i])
            rewards[:, i] = ri
        return rewards

    def _get_one(self, state_vec: np.ndarray, dones: np.ndarray) -> np.ndarray:
        dones = dones.astype(int)
        idx, = np.where(dones == 1)
        rewards = deque()
        idx = idx.tolist() + [len(dones) - 1]
        i = 0
        for j in idx:
            trace = pd.DataFrame(
                state_vec[i: j + 1],
                columns=self.signals,
                index=np.arange(i, j + 1) * self.dt,
            )
            ri = self.rho(self.phi, trace)  # type: np.ndarray
            rewards.extend(ri.tolist())
            i = j + 1
            if len(rewards) == len(dones):
                break
        return rewards
