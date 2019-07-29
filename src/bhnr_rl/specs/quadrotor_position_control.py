from math import pi as PI

import numpy as np
import sympy as sp

import temporal_logic.signal_tl as stl

SIGNALS = (
    'x_d', 'y_d', 'z_d',
    'roll', 'pitch', 'yaw',
    'dot_x', 'dot_y', 'dot_z',
    'dot_roll', 'dot_pitch', 'dot_yaw',
    'x_g', 'y_g', 'z_g',
    'collision'
)

dt = 50e-3  # s

(x_d, y_d, z_d,
 roll_d, pitch_d, yaw_d,
 dotx, doty, dotz,
 dotax, dotay, dotaz,
 x_g, y_g, z_g,
 collision) = stl.signals(SIGNALS)

drone_pos = sp.Matrix([x_d, y_d, z_d])
drone_ori = sp.Matrix([roll_d, pitch_d, yaw_d])
drone_lv = sp.Matrix([dotx, doty, dotz])
drone_av = sp.Matrix([dotax, dotay, dotaz])

goal_pos = sp.Matrix([x_g, y_g, z_g])

POSITION_SPEC = stl.Predicate((goal_pos - drone_pos).norm() <= 0.01)

GOAL_VELOCITY_SPEC = POSITION_SPEC >> stl.Predicate(
    drone_lv.norm() <= 0.001)  # Reached goal => stay still

# Keep roll and pitch within 30deg
ANGLE_CONSTRAINT = (
    stl.Predicate(abs(roll_d) <= PI/6)
    & stl.Predicate(abs(pitch_d) <= PI/6)
)

# Minimize magnitude of angular velocity to <= 5deg/s
ANGULAR_VEL_CONSTRAINT = stl.Predicate(drone_av.norm() <= np.deg2rad(5))


# SPEC = stl.G(                   # Always do the following:
#     stl.F(POSITION_SPEC)        # Head towards goal position
#     & ANGLE_CONSTRAINT          # Keep the angles constrained
#     & ANGULAR_VEL_CONSTRAINT    # COnstrain the angular velocity
#     & GOAL_VELOCITY_SPEC        # Minimize drift once you reach goal
#     & (1000 * collision < 1000)
# )

PHI = stl.F(POSITION_SPEC, (0, 5))  # Reach goal position within 5 sec
PHI_SAFE = stl.G(PHI & ANGULAR_VEL_CONSTRAINT)

SPEC = PHI_SAFE

SPEC1 = PHI
