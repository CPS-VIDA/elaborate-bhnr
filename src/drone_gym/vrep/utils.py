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

import logging
import subprocess as sp


log = logging.getLogger('VREP.API')


class _ProcInstance:
    def __init__(self, args, log_file=None):
        self.args = args
        self.inst = None

        self.log_file = log_file

    def start(self):
        log.info('Starting V-REP Instance...')
        log.debug(' '.join(self.args))
        try:
            with open(self.log_file, 'w') as log_file:
                self.inst = sp.Popen(self.args, stdout=log_file, stderr=sp.STDOUT)
        except EnvironmentError:
            log.error('Launching Instance, cannot find executable at {}'.format(self.args[0]))
            raise

        return self

    def is_alive(self):
        return True if self.inst and self.inst.poll() is None else False

    def end(self):
        log.info('Terminating V-REP Instance...')
        if self.is_alive():
            self.inst.terminate()
            retcode = self.inst.wait()
        else:
            retcode = self.inst.returncode
        log.info('V-REP Instance exited with code: {}'.format(retcode))
        return self


def check_ret(ret_tuple, ignore_one=False):
    """
    check return tuple, raise error if retcode is not OK,
    return remaining data otherwise
    :param ret_tuple:
    :param ignore_one:
    :return:
    """
    istuple = isinstance(ret_tuple, tuple)
    if not istuple:
        ret = ret_tuple
    else:
        ret = ret_tuple[0]

    if (not ignore_one and ret != vrep.simx_return_ok) or (ignore_one and ret > 1):
        raise RuntimeError('Return code(' + str(ret) + ') not OK, API call failed. Check the parameters!')

    return ret_tuple[1:] if istuple else None


class SimOpModes:
    # TODO: Fill out the OpModes?
    oneshot = vrep.simx_opmode_oneshot  #: Send/Recv 1 chunk (Async)
    blocking = vrep.simx_opmode_blocking  #: Send/Recv 1 chunk (Sync)

    oneshot_wait = vrep.simx_opmode_oneshot_wait
    continuous = vrep.simx_opmode_continuous
    streaming = vrep.simx_opmode_streaming

    oneshot_split = vrep.simx_opmode_oneshot_split
    continuous_split = vrep.simx_opmode_continuous_split
    streaming_split = vrep.simx_opmode_streaming_split

    # Special operation modes
    discontinue = vrep.simx_opmode_discontinue
    buffer = vrep.simx_opmode_buffer
    remove = vrep.simx_opmode_remove

class GUIItems:
    menubar = 0x0001
    popups = 0x0002
    toolbar1 = 0x0004
    toolbar2 = 0x0008
    hierarchy = 0x0010
    infobar = 0x0020
    statusbar = 0x0040
    scripteditor = 0x0080
    scriptsimulationparameters = 0x0100
    dialogs = 0x0200
    browser = 0x0400
    all_elements = 0xffff
