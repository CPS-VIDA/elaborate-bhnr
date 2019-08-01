import argparse
import logging
import multiprocessing
import os
import shutil
from datetime import datetime
import atexit

import gym
import drone_gym
from stable_baselines import PPO2
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder

from bhnr_rl.specs.register import get_spec
from bhnr_rl.tl.monitor import STLMonitor
from bhnr_rl.tl.tl_ppo import PPO2 as TLPPO2

from sympy import latex


log = logging.getLogger()


@atexit.register
def kill_vrep():
    import subprocess
    import signal

    log.info('Killing all V-REP instances')
    subprocess.run('pkill -9 vrep', shell=True)


def parse_args():
    parser = argparse.ArgumentParser('Run PPO2')
    parser.add_argument('--env-id', type=str,
                        default='SimpleQuadrotorPositionControlEnv-v0')
    parser.add_argument('--n-cpu', type=int, help='Number of CPUs to use',
                        default=multiprocessing.cpu_count() // 2)
    parser.add_argument('--n-steps', type=int,
                        help='Number of rollout steps', default=2048)
    parser.add_argument('--total-steps', type=float,
                        help='Total number of steps', default=2e6)
    parser.add_argument('--log-dir', type=str,
                        help='Logging dir', default='log')
    parser.add_argument(
        '--seed', help='Random seed to init with', nargs="+", type=int)
    parser.add_argument('--use-stl', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--video-dir', type=str, default='videos')
    return parser.parse_args()


def create_env(env_id, _seed, rank, _log_dir):
    _env = gym.make(env_id)
    _seed = 0 if seed is None else seed
    _env.seed(_seed * 100 + rank)
    _env = Monitor(_env, log_dir, True)
    return _env


def gen_env(env_id, seed, log_dir, n_cpu, video_dir, now, record):
    env = DummyVecEnv([lambda: create_env(env_id, seed, i, log_dir)
                       for i in range(n_cpu)])
    if record:
        env = VecVideoRecorder(env,
                               video_folder=video_dir,
                               record_video_trigger=lambda x: x % 1000 == 0,
                               name_prefix='{}'.format(now),
                               video_length=2000)
    return env


def gen_model(env_id, env, log_dir, verbose, n_steps, use_stl):
    if use_stl:
        model = TLPPO2(MlpPolicy, env, verbose=True,
                       n_steps=n_steps, tensorboard_log=log_dir)
        spec, signals, monitor = get_spec(env_id)
        model.monitor = STLMonitor(spec, signals, monitor, n_steps, 5e-2)
        log.info('STL: {}'.format(latex(spec)))
    else:
        model = PPO2(MlpPolicy, env, verbose=verbose,
                     n_steps=n_steps, tensorboard_log=log_dir)
    return model


if __name__ == '__main__':
    args = parse_args()
    print(args)
    env_id = args.env_id
    n_cpu = args.n_cpu
    n_steps = args.n_steps
    total_timesteps = int(args.total_steps)
    log_dir = args.log_dir
    verbose = int(args.verbose)
    os.makedirs(log_dir, exist_ok=True)
    seeds = [None] if not args.seed else args.seed
    # seed = int(args.seed) if args.seed is not None else None
    name = 'TLPPO' if args.use_stl else 'PPO'

    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    record = args.record
    video_dir = os.path.join(args.video_dir, env_id, name)
    os.makedirs(video_dir, exist_ok=True)

    for seed in seeds:
        successful_sess = False
        while not successful_sess:
            try:
                now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                session_log_dir = '{}/seed-{}/'.format(log_dir, seed)
                os.makedirs(session_log_dir, exist_ok=True)
                env = gen_env(env_id, seed, session_log_dir,
                              n_cpu, video_dir, now, record)
                model = gen_model(env_id, env, session_log_dir,
                                  verbose, n_steps, args.use_stl)
                set_global_seeds(seed)

                model.learn(total_timesteps, reset_num_timesteps=True)
            except RuntimeError:
                # Delete the log dir
                shutil.rmtree(session_log_dir, ignore_errors=True)
            else:
                successful_sess = True
                backup_dir = os.path.join('backup', name)
                os.makedirs(backup_dir, exist_ok=True)
                model.save(os.path.join(
                    backup_dir, '{}-{}-{}'.format(name, seed, now)))
            kill_vrep()
