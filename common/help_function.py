import os
import gym
import torch
import random
import shutil
import numpy as np


def set_random_seed(env: gym.Env, seed: int):
    """Set random seed"""
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def check_path_or_make_dir(path):
    dir_name = os.path.dirname(path)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

def remove_dir(dir_name):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
