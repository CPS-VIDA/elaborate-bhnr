"""Wrapper around V-REP Remote API

A lot of this is adapted from https://github.com/ctmakro/vrepper/

"""

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

import atexit
import inspect
import os
import random
import shutil
import types
from collections import deque

from .utils import check_ret, _ProcInstance, SimOpModes, log
from .vrep_object import VREPObject

from datetime import datetime

PROC_LIST = deque()


@atexit.register
def cleanup():
    global PROC_LIST
    for p in PROC_LIST:  # type: _ProcInstance
        p.end()


class VREPSim:

    def __init__(self,
                 port_num=None, debug=False, sync=True,
                 headless=False,
                 start_auto=False, sim_duration=None, quit_on_complete=False,
                 addon1=None, addon2=None,
                 scene=None, model=None,
                 gui_elements_disable=None,
                 ):
        if port_num is None:
            port_num = int(random.random() * 1000 + 19999)
        self.port_num = port_num

        vrep_exec = shutil.which('vrep.sh', os.X_OK)
        if vrep_exec is None:
            vrep_exec = shutil.which('vrep', os.X_OK)
        if vrep_exec is None:
            log.error('Unable to find V-REP executable in env PATH')
            raise RuntimeError(
                'V-REP executable not found: vrep.sh, vrep, vrep.exe')
        log.info('Using V-REP executable: {}'.format(vrep_exec))

        launch_args = [vrep_exec]
        if headless:
            launch_args.append('-h')
        if start_auto:
            if not sim_duration:
                raise ValueError(
                    'Given auto start, but not how long sim needs to run')
            launch_args.append('-g{}'.format(sim_duration))
        if quit_on_complete:
            launch_args.append('-q')
        if addon1:
            launch_args.append('-a{}'.format(addon1))
        if addon2:
            launch_args.append('-a{}'.format(addon1))

        launch_args.append(
            '-gREMOTEAPISERVERSERVICE_{}_{}_{}'.format(port_num, str(debug).upper(), str(sync).upper()))
        if gui_elements_disable:
            launch_args.append(
                '-gGUIITEMS_{:d}'.format(int(gui_elements_disable))
            )
        if scene:
            launch_args.append('{}'.format(scene))
        if model:
            launch_args.append('{}'.format(model))

        log.info('CMD: {}'.format(' '.join(launch_args)))

        # A reference to the instance of the V-REP sim
        self.launch_args = launch_args
        log_dir = os.path.abspath('./log')
        os.makedirs(log_dir, exist_ok=True)
        log_file = 'vrep_{}_{}.log'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'), port_num)
        log.info('Logging to: {}'.format(log_file))
        self.instance = _ProcInstance(launch_args, os.path.join(log_dir, log_file))

        # clientID of the instance when connected to server,
        # to differentiate between instances in the driver
        self.client_id = -1

        self.started = False
        self.sim_running = False

        # assign every API function call from vrep to self
        vrep_methods = [a for a in dir(vrep) if
                        not a.startswith('__') and isinstance(getattr(vrep, a), types.FunctionType)]

        def assign_from_vrep_to_self(_name):
            wrapped = getattr(vrep, _name)
            sign = inspect.signature(wrapped)
            if 'clientID' in sign.parameters:
                def func(*args, **kwargs):
                    return wrapped(self.client_id, *args, **kwargs)
            else:
                def func(*args, **kwargs):
                    return wrapped(*args, **kwargs)
            setattr(self, _name, func)

        for name in vrep_methods:
            assign_from_vrep_to_self(name)

        PROC_LIST.append(self)

    def start(self):
        if self.started:
            log.error('V-REP Instance has already been started...')
            raise RuntimeError(
                'You are calling start on this V-REP instance twice. Check your script')

        log.info('Attempting to start V-REP instance')
        self.instance.start()

        retries = 0
        connected = False
        while not connected:
            log.info('Trying to connect to server on 127.0.0.1:{} [Attempt: {:d}]'.format(
                self.port_num, retries + 1))
            # vrep.simxFinish(-1) # just in case, close all opened connections

            self.client_id = self.simxStart(
                '127.0.0.1', self.port_num,
                waitUntilConnected=True,
                doNotReconnectOnceDisconnected=True,
                timeOutInMs=1000,
                commThreadCycleInMs=0
            )
            connected = (self.client_id != -1)
            retries += 1
            if retries >= 15 and not connected:
                self.end()
                raise RuntimeError(
                    'Unable to connect to V-REP after 15 attempts. Socket Address: 127.0.0.1:{}'.format(
                        self.port_num)
                )

        log.info('Connected to V-REP Instance at 127.0.0.1:{}'.format(self.port_num))

        objs, = check_ret(self.simxGetObjects(
            vrep.sim_handle_all, SimOpModes.blocking))
        log.info('Number of objects in scene: ' + str(len(objs)))

        # Send some non-blocking data to V-REP
        self.simxAddStatusbarMessage('Hello V-REP!', SimOpModes.oneshot)

        # Setup a useless signal
        self.simxSetIntegerSignal('asdf', 1, SimOpModes.blocking)

        log.info(
            'V-REP instance started, remote API connection created. Everything seems to be ready.')

        self.started = True

        return self

    def end(self):
        log.info('V-REP Instance shutting down: {}'.format(self.sim_running))
        if self.sim_running:
            self.stop_simulation()
        self.simxFinish()
        self.instance.end()

        self.started = False

        log.info('V-REP Instance has been shut down')
        return self

    def reset_toggle_headless(self):
        if '-h' in self.launch_args:
            self.launch_args.remove('-h')
        else:
            self.launch_args.insert(0, '-h')

    def load_scene(self, path_to_scene):
        log.info('Loading scene from {} in server'.format(path_to_scene))
        try:
            check_ret(self.simxLoadScene(
                path_to_scene, 0, SimOpModes.blocking))
        except Exception:
            log.error('Scene loading failure')
            raise

    def start_blocking_simulation(self):
        self.start_simulation(True)

    def start_async_simulation(self):
        self.start_simulation(False)

    def start_simulation(self, is_sync):
        # IMPORTANT
        # you should poll the server state to make sure
        # the simulation completely stops before starting a new one
        log.debug('Polling V-REP to check if it is still running')
        while True:
            # poll the useless signal (to receive a message from server)
            check_ret(self.simxGetIntegerSignal('asdf', SimOpModes.blocking))

            # check server state (within the received message)
            e = self.simxGetInMessageInfo(vrep.simx_headeroffset_server_state)

            # check bit0
            not_stopped = e[1] & 1

            if not not_stopped:
                break

        # enter sync mode
        log.debug('Starting simulation. SYNC: {}'.format(is_sync))
        check_ret(self.simxSynchronous(is_sync))
        check_ret(self.simxStartSimulation(SimOpModes.blocking))
        self.sim_running = True

    def make_simulation_synchronous(self, sync=True):
        if not self.sim_running:
            log.info('Simulation doesn\'t seem to be running. Starting up')
            self.start_simulation(sync)
        else:
            check_ret(self.simxSynchronous(sync))

    def stop_simulation(self):
        log.info('Stopping V-REP simulation')
        check_ret(self.simxStopSimulation(SimOpModes.oneshot), ignore_one=True)
        self.sim_running = False

    def step_blocking_simulation(self):
        check_ret(self.simxSynchronousTrigger())

    def get_objects(self):
        return check_ret(self.simxGetObjects(vrep.sim_handle_all, SimOpModes.blocking))

    def get_object_handle(self, name):
        handle, = check_ret(self.simxGetObjectHandle(
            name, SimOpModes.blocking))
        return handle

    def get_object_by_handle(self, handle, is_joint=True):
        """
        Get the vrep object for a given handle
        :param int handle: handle code
        :param bool is_joint: True if the object is a joint that can be moved
        :returns: vrepobject
        """
        return VREPObject(self, handle, is_joint)

    def get_object_by_name(self, name, is_joint=True):
        """
        Get the vrep object for a given name
        :param str name: name of the object
        :param bool is_joint: True if the object is a joint that can be moved
        :returns: VREPObject
        """
        return self.get_object_by_handle(self.get_object_handle(name), is_joint)

    @staticmethod
    def create_params(ints=[], floats=[], strings=[], bytes=''):
        if bytes == '':
            bytes_in = bytearray()
        else:
            bytes_in = bytes
        return ints, floats, strings, bytes_in

    def call_script_function(self, function_name, params, script_name="remoteApiCommandServer"):
        """
        Calls a function in a script that is mounted as child in the scene
        :param str script_name: the name of the script that contains the function
        :param str function_name: the name of the function to call
        :param tuple params: the parameters to call the function with (must be 4 parameters: list of integers, list of
                floats, list of string, and bytearray
        :returns: tuple (res_ints, res_floats, res_strs, res_bytes)
            WHERE
            list res_ints is a list of integer results
            list res_floats is a list of floating point results
            list res_strs is a list of string results
            bytearray res_bytes is a bytearray containing the resulting bytes
        """
        assert type(params) is tuple
        assert len(params) == 4

        return check_ret(self.simxCallScriptFunction(
            script_name,
            vrep.sim_scripttype_childscript,
            function_name,
            params[0],  # integers
            params[1],  # floats
            params[2],  # strings
            params[3],  # bytes
            SimOpModes.blocking
        ))

    def set_signal(self, signal_name, signal_val):
        if isinstance(signal_val, str):
            return check_ret(
                self.simxSetStringSignal(
                    signal_name,
                    signal_val,
                    SimOpModes.oneshot,
                ),
                ignore_one=True
            )
        if isinstance(signal_val, int):
            return check_ret(
                self.simxSetIntegerSignal(
                    signal_name,
                    signal_val,
                    SimOpModes.oneshot,
                ),
                ignore_one=True
            )
        if isinstance(signal_val, float):
            return check_ret(
                self.simxSetFloatSignal(
                    signal_name,
                    signal_val,
                    SimOpModes.oneshot,
                ),
                ignore_one=True
            )
        raise ValueError(
            'Unsopported signal type: {}'.format(type(signal_val)))

    def get_signal(self, signal_name, signal_type):
        if signal_type == str:
            return check_ret(
                self.simxGetStringSignal(
                    signal_name,
                    SimOpModes.blocking,
                )
            )
        if signal_type == int:
            return check_ret(
                self.simxGetIntegerSignal(
                    signal_name,
                    SimOpModes.blocking,
                )
            )
        if signal_type == float:
            return check_ret(
                self.simxGetFloatSignal(
                    signal_name,
                    SimOpModes.blocking,
                )
            )
        raise ValueError('Unsopported signal type: {}'.format(signal_type))
