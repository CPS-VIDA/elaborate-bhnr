import temporal_logic.signal_tl as stl
from typing import Callable, Tuple, Any
import pandas as pd

SPEC_REGISTRY = dict()

STLMonitorType = Callable[[stl.Expression, pd.DataFrame, Any], pd.Series]


def register(env_id: str, spec: stl.Expression, signals: Tuple[str, ...],
             monitor: STLMonitorType):
    SPEC_REGISTRY[env_id] = (spec, signals, monitor)


def get_spec(env_id: str) -> Tuple[stl.Expression, Tuple[str, ...], STLMonitorType]:
    if env_id in SPEC_REGISTRY:
        return SPEC_REGISTRY[env_id]
    raise ValueError('Given env id not found in registry: {}'.format(env_id))
