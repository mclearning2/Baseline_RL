import os
import wandb
import torch
import shutil
import pickle
import numpy as np
from glob import glob

from common.logger import logger

def restore_wandb(
    user_name: str,
    project: str,
    run_id: str,
    params_path: str,
    hyperparams_path: str
    ):
    """ wandb에서 파라미터와 하이퍼파라미터를 복원 """
    
    run_path = os.path.join(user_name, project, run_id)

    for path in [params_path, hyperparams_path]:
        
        root = os.path.dirname(path)
        name = os.path.basename(path)
        
        downloaded = wandb.restore(
            name=name,
            run_path=run_path,
            replace=True,
            root=root)

def restore_hyperparams(hyperparams_path):
    ''' pickle 형식(.pkl)으로 저장했던 hyperparameter를 다시 불러온다. '''

    with open(hyperparams_path, 'rb') as f:
        unpickler = pickle.Unpickler(f)
        hyperparams = unpickler.load()

    return hyperparams

def save_hyperparams(hyperparams, hyperparams_path):
    check_path_and_make_dir(hyperparams_path)

    with open(hyperparams_path, 'wb+') as f:
        pickle.dump(hyperparams, f)

def restore_model_params(model, params_path):
    params = torch.load(params_path)
    for name, tensor in model.items():
        model[name].load_state_dict(params[name])

def save_model_params(model, params_path):
    check_path_and_make_dir(params_path)

    params = dict()
    for name, tensor in model.items():
        params[name] = model[name].state_dict()

    torch.save(params, params_path)

    logger.info("Saved model and optimizer to " + params_path)

def save_wandb(params_path, hyperparams_path, video_dir):
    wandb.save(params_path)
    wandb.save(hyperparams_path)
    files = glob(os.path.join(video_dir, "*.mp4"))
    for mp4_file in files:
        wandb.save(mp4_file)
        
def check_path_and_make_dir(path: str):
    dir_name = os.path.dirname(path)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)

def remove_dir(dir_name: str):
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)
