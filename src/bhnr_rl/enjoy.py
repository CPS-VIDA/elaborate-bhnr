import argparse
import logging
import multiprocessing
import os
from datetime import datetime
import pathlib
import random
import atexit
import shutil

import pandas as pd
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
import temporal_logic.signal_tl as STL
from bhnr_rl.specs import quadrotor_position_control as QUAD

log = logging.getLogger()


ANGLE_SPEC = STL.G(QUAD.ANGLE_CONSTRAINT)
POSITION_SPEC = STL.G(QUAD.PHI)


def parse_args():
    parser = argparse.ArgumentParser('Run PPO2')
    parser.add_argument('--env-id', type=str,
                        default='SimpleQuadrotorPositionControlEnv-v0')
    parser.add_argument('--log-dir', type=str,
                        help='Logging dir', default='log')
    parser.add_argument('--backup-dir', type=str,
                        help='DIrectory with trained models', required=True)
    parser.add_argument(
        '--seed', help='Random seed to init with', nargs='+', type=int, default=[None])
    parser.add_argument(
        '--n-runs', help='Number of eval runs', type=int, default=1)
    parser.add_argument(
        '--n-steps', help='Number of steps in eval run', type=int, default=200
    )
    parser.add_argument('--use-stl', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--video-dir', type=str, default='videos')
    return parser.parse_args()


@atexit.register
def kill_vrep():
    import subprocess
    import signal

    log.info('Killing all V-REP instances')
    subprocess.run('pkill -9 vrep', shell=True)


def create_env(env_id, seed, rank, log_dir):
    _env = gym.make(env_id)
    _seed = 0 if seed is None else seed
    _env.seed(_seed * 100 + rank)
    _env = Monitor(_env, log_dir, True)
    return _env


def gen_env(env_id, seed, log_dir, record, video_dir):
    env = DummyVecEnv([lambda: create_env(env_id, seed, 0, log_dir)])
    if record:
        now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        env = VecVideoRecorder(env,
                               video_folder=video_dir,
                               record_video_trigger=lambda x: x % 1000 == 0,
                               name_prefix='{}'.format(now))
    return env


def load_model(model_path, use_stl):
    if use_stl:
        return TLPPO2.load(model_path)
    else:
        return PPO2.load(model_path)


def eval_run(env, model, n_steps, log_dir):
    from collections import deque
    states = deque()

    obs = env.reset()
    log.info(obs.shape)
    for i in range(n_steps):
        states.append(obs.flatten())
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)

    return states


def save_states(states, log_dir, headers=None):
    state_file = pathlib.Path(log_dir) / 'trace.csv'
    state_df = pd.DataFrame(states, columns=headers)
    state_df.to_csv(state_file)
    return state_df


def main():
    args = parse_args()

    env_id = args.env_id
    log_dir = args.log_dir
    backup_dir = args.backup_dir
    seeds = args.seed
    n_runs = args.n_runs
    n_steps = args.n_steps
    use_stl = args.use_stl
    verbose = args.verbose
    record = args.record
    video_dir = args.video_dir
    name = "TLPPO" if use_stl else "PPO"

    logging.basicConfig(level=logging.INFO)

    if len(seeds) > 1:
        assert (len(
            seeds) == n_runs), "Number of seeds must equal number or runs, or set common (one) seed"
    else:
        seeds = seeds * n_runs

    # Get all checkpoints avaliable
    checkpoints_dir = pathlib.Path(backup_dir)
    log.info("Checkpoints in dir: {}".format(checkpoints_dir))
    checkpoints = list(checkpoints_dir.glob(name + '*.pkl'))
    log.info("Checkpoints available: {}".format(checkpoints))
    assert (len(checkpoints) > 0)

    for run in range(n_runs):
        seed = seeds[run]
        set_global_seeds(seed)

        model_path = str(random.sample(checkpoints, 1)[0])
        log.info('Attempting to load: {}'.format(model_path))
        successful_sess = False
        while not successful_sess:
            try:
                now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                session_log_dir = '{}/seed-{}/'.format(log_dir, seed)
                os.makedirs(session_log_dir, exist_ok=True)

                env = gen_env(env_id, seed, session_log_dir, record, video_dir)
                model = load_model(model_path, use_stl)

                states = eval_run(env, model, n_steps, session_log_dir)
            except RuntimeError:
                # Delete the log dir
                shutil.rmtree(session_log_dir, ignore_errors=True)
            else:
                successful_sess = True
                spec, signals, monitor = get_spec(env_id)
                state_df = save_states(states, session_log_dir, signals)
                robustness1 = monitor(POSITION_SPEC, state_df)
                robustness2 = monitor(ANGLE_SPEC, state_df)
                log.info("Robustness: POSITION = {} | ANGLE = {}".format(
                    robustness1.iloc[0], robustness2.iloc[0]))
                robustness_df = pd.DataFrame(
                    dict(position_spec=robustness1, angle_spec=robustness2))
                robustness_df.to_csv(os.path.join(
                    session_log_dir, 'robustness.csv'))


if __name__ == '__main__':
    main()
