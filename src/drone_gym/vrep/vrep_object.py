# import the vrep library
try:
    from . import vrep
except Exception:
    print('--------------------------------------------------------------')
    print('"vrep.py" could not be imported. This means very probably that')
    print('either "vrep.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "vrep.py"')
    print('--------------------------------------------------------------')
    print('')
    raise

import numpy as np
from numpy import deg2rad, rad2deg

from .utils import check_ret, SimOpModes

blocking = SimOpModes.blocking


class VREPObject:
    def __init__(self, env, handle, is_joint=True):
        self.env = env
        self.handle = handle
        self.is_joint = is_joint

        # Flags if values have been read in streaming mode
        self.str_ori = 0
        self.str_pos = 0
        self.str_vel = 0

    def get_orientation(self, relative_to=None, stream=False):
        mode = blocking
        ignore_one = False
        if stream:
            if self.str_ori > 0:
                mode = SimOpModes.buffer
            else:
                mode = SimOpModes.streaming
            ignore_one = self.str_ori <= 1
            self.str_ori += 1
        eulerAngles, = check_ret(self.env.simxGetObjectOrientation(
            self.handle,
            -1 if relative_to is None else relative_to.handle,
            mode), ignore_one)
        return eulerAngles

    def set_orientation(self, x, y, z, relative_to=None):
        return check_ret(self.env.simxSetObjectOrientation(
            self.handle,
            -1 if relative_to is None else relative_to.handle,
            (x, y, z),
            blocking
        ))

    def get_position(self, relative_to=None, stream=False):
        mode = blocking
        ignore_one = False
        if stream:
            if self.str_pos > 0:
                mode = SimOpModes.buffer
            else:
                mode = SimOpModes.streaming
            ignore_one = self.str_pos <= 1
            self.str_pos += 1
        position, = check_ret(self.env.simxGetObjectPosition(
            self.handle,
            -1 if relative_to is None else relative_to.handle,
            mode), ignore_one)
        return position

    def set_position(self, x, y, z, relative_to=None):
        return check_ret(self.env.simxSetObjectPosition(
            self.handle, -1 if relative_to is None else relative_to.handle, [x, y, z], blocking))

    def get_velocity(self, stream=False):
        mode = blocking
        ignore_one = False
        if stream:
            if self.str_vel > 0:
                mode = SimOpModes.buffer
            else:
                mode = SimOpModes.streaming
            ignore_one = self.str_vel <= 1
            self.str_vel += 1
        return check_ret(self.env.simxGetObjectVelocity(
            self.handle,
            # -1 if relative_to is None else relative_to.handle,
            mode), ignore_one)

    def set_joint_velocity(self, v):
        self._check_joint()
        return check_ret(self.env.simxSetJointTargetVelocity(
            self.handle,
            v,
            blocking))

    def set_joint_force(self, f):
        self._check_joint()
        return check_ret(self.env.simxSetJointForce(
            self.handle,
            f,
            blocking))

    def set_joint_position_target(self, angle):
        """
        Set desired position of a servo
        :param int angle: target servo angle in degrees
        :return: None if successful, otherwise raises exception
        """
        self._check_joint()
        return check_ret(self.env.simxSetJointTargetPosition(
            self.handle,
            -deg2rad(angle),
            blocking))

    def get_joint_angle(self, stream=False):
        self._check_joint()
        angle = check_ret(
            self.env.simxGetJointPosition(
                self.handle,
                blocking
            )
        )
        return -rad2deg(angle[0])

    def get_joint_force(self, stream=False):
        self._check_joint()
        force = check_ret(
            self.env.simxGetJointForce(
                self.handle,
                blocking
            )
        )
        return force

    def read_force_sensor(self):
        state, forceVector, torqueVector = check_ret(self.env.simxReadForceSensor(
            self.handle,
            blocking))

        if state & 1 == 1:
            return None  # sensor data not ready
        else:
            return forceVector, torqueVector

    def get_vision_image(self):
        resolution, image = check_ret(self.env.simxGetVisionSensorImage(
            self.handle,
            0,  # options=0 -> RGB
            blocking,
        ))
        dim, im = resolution, image
        nim = np.array(im, dtype='uint8')
        nim = np.reshape(nim, (dim[1], dim[0], 3))
        nim = np.flip(nim, 0)  # LR flip
        nim = np.flip(nim, 2)  # RGB -> BGR
        return nim

    def _check_joint(self):
        if not self.is_joint:
            raise Exception(
                "Trying to call a joint function on a non-joint object.")
