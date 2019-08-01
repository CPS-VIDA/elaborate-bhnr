from .register import register, get_spec

from temporal_logic.signal_tl.monitors import lti_filter_monitor, efficient_robustness

from . import quadrotor_position_control

register('SimpleQuadrotorPositionControlEnv-v0',
         quadrotor_position_control.SPEC,
         quadrotor_position_control.SIGNALS,
         efficient_robustness)

register('VREPQuadrotorPositionControlEnv-v0',
         quadrotor_position_control.SPEC,
         quadrotor_position_control.SIGNALS,
         efficient_robustness)

register('QuadrotorAngularConstraint',
         quadrotor_position_control.SPEC,
         quadrotor_position_control.SIGNALS,
         efficient_robustness)

register('QuadrotorAngularFiltering',
         quadrotor_position_control.SPEC,
         quadrotor_position_control.SIGNALS,
         lti_filter_monitor)
