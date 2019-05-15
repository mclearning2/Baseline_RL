import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from common.envs.gym import Gym
from common.abstract.base_project import BaseProject
from common.models.mlp import NormalDist, MLP, SepActorCritic
from algorithms.PPO import PPO

class Project(BaseProject):
    def init_hyper_params(self):
        return {
            "gamma": 0.99,
            "tau": 0.95,
            "epsilon": 0.3105,
            "entropy_ratio": 0.001,
            "rollout_len": 30,
            "batch_size": 87,
            "epoch": 22,
            
            "n_workers": 10,
            "max_episode_steps": 0,
            "actor_lr": 0.005262,
            "critic_lr": 0.008651,
            "actor_hidden_sizes": [],
            "critic_hidden_sizes": [29],
        }

    def init_env(self, hyper_params, monitor_func):
        return Gym(
            env_id='BipedalWalker-v2', 
            n_envs=hyper_params['n_workers'],
            max_episode= 300,
            max_episode_steps=hyper_params['max_episode_steps'],
            monitor_func=monitor_func(lambda x: x % 50 == 0),
            scale_action=True,
        )

    def init_model(self, input_size, output_size, device, hyper_params):

        actor = NormalDist(
            input_size=input_size,
            hidden_sizes=hyper_params['actor_hidden_sizes'],
            output_size=output_size,
            output_activation=nn.Tanh(),
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