import torch.nn as nn
import torch.optim as optim

from gym.gym import Gym
from common.abstract.base_project import BaseProject
from algorithms.models.mlp import CategoricalDist, MLP, SepActorCritic
from algorithms.PPO import PPO

class Project(BaseProject):
    def init_hyper_params(self):

        return {
            "gamma": 0.99,
            "tau": 0.95,
            "epsilon": 0.3881,
            "entropy_ratio": 0.001,
            "rollout_len": 5,
            "batch_size": 5,
            "epoch": 8,
            "n_workers": 5,
            "max_episode_steps": 0,
            "actor_lr": 0.007007,
            "critic_lr": 0.003648,
            "actor_hidden_sizes": [],
            "critic_hidden_sizes": [70],
        }
    
    def init_env(self, hyper_params, monitor_func):
        return Gym(
            env_id='Acrobot-v1', 
            n_envs=hyper_params['n_workers'],
            max_episode= 300,
            max_episode_steps=hyper_params['max_episode_steps'],
            monitor_func=monitor_func(lambda x: x % 50 == 0),
        )

    def init_model(self, input_size, output_size, device, hyper_params):

        actor = CategoricalDist(
            input_size=input_size,
            hidden_sizes=hyper_params['actor_hidden_sizes'],
            output_size=output_size
        )

        critic = MLP(
            input_size=input_size,
            hidden_sizes=hyper_params['critic_hidden_sizes'],
            output_size=1
        )

        model = SepActorCritic(actor, critic).to(device)

        optimizer = optim.Adam(model.parameters(), hyper_params['actor_lr'])

        return {"model": model, "optim": optimizer}

    def init_agent(self, env, models, device, hyper_params):
        return PPO(env, models['model'], models['optim'], device, hyper_params)