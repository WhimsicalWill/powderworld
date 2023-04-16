import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import torch.nn as nn
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder

from powderworld import PWSim
from powderworld.envs import PWSandEnv, PWDestroyEnv, PWDrawEnv, PWCreativeEnv

from PIL import Image


def train(run_name, env_name, kwargs_pcg, args):
    print("env name is", env_name)
    env_fn = None
    if env_name == 'sand':
        env_fn = PWSandEnv
    elif env_name == 'draw':
        env_fn = PWDrawEnv
    elif env_name == 'destroy':
        env_fn = PWDestroyEnv
    elif env_name == 'creative':
        env_fn = PWCreativeEnv

    env = env_fn(test=False, kwargs_pcg=kwargs_pcg, device='cuda')
    env = VecMonitor(env)
    env = VecVideoRecorder(env, "videos/random_agent", record_video_trigger=lambda x: True, video_length=200)

    total_timesteps = 1_000
    obs = env.reset()
    save_goal_image(env)
    steps = 0
    no_op = np.array([0, 0, 0, 0, 0])
    for _ in range(total_timesteps):
        if steps < 10:
            action = env.action_space.sample()
        else:
            action = no_op
        observation, reward, done, info = env.step(action)
        steps += 1
        if done.any():
            env.reset()
            save_goal_image(env)
            steps = 0

def save_goal_image(env, env_id=0):
    im = env.pwr.render(env.goal_state[[env_id]])
    im = Image.fromarray(im)
    im = im.resize((256, 256), Image.NEAREST)
    num = np.random.randint(0, 100000)
    im.save(f"goal_test/goal_{num}.png")


if __name__ == "__main__":
    print("Loading args")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir')
    parser.add_argument('--num_elems', type=int, default=6)
    parser.add_argument('--num_lines', type=int, default=2)
    parser.add_argument('--num_circles', type=int, default=2)
    parser.add_argument('--num_squares', type=int, default=2)
    parser.add_argument('--num_tasks', type=int, default=100000)
    parser.add_argument('--env_name')

    args = parser.parse_args()
    print(args)

    if args.env_name is None:
        raise Exception("Provide an environment to train on !")

    all_elems = ['empty', 'sand', 'water', 'wall', 'plant', 'fire', 'wood', 'ice', 'lava', 'dust', 'cloner', 'gas', 'acid', 'stone']
    elems = all_elems[:args.num_elems]
    kwargs_pcg = dict(hw=(64,64), elems=elems, num_tasks=args.num_tasks, num_lines=args.num_lines,
                      num_circles=args.num_circles, num_squares=args.num_squares)
    train(args.savedir, args.env_name, kwargs_pcg, args)
