import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim

from common.envs.core import GymEnv
from common.abstract.base_project import BaseProject

from common.models.mlp import NormalDistMLP, MLP
from algorithms.PPO import PPO

class Project(BaseProject):
    def init_hyper_params(self):
        return {
            "gamma": 0.9204,
            "tau": 0.9094,
            "epsilon": 0.389,
            "entropy_ratio": 0.001,
            "rollout_len": 14,
            "mini_batch_size": 92,
            "epoch": 17,
            "n_workers": 8,
            "max_episode_steps": 0,
            "actor_lr": 0.001238,
            "critic_lr": 0.00215,
            "actor_hidden_sizes": [41],
            "critic_hidden_sizes": [61, 61],
        }
    
    def init_env(self, hyper_params, render_on, monitor_func):
        return GymEnv(
            env_id = 'Pendulum-v0', 
            n_envs = hyper_params['n_workers'],
            render_on = render_on,
            max_episode = 500,
            max_episode_steps = hyper_params['max_episode_steps'],
            video_call_func = monitor_func(lambda x: x % 50 == 0)
        )

    def init_models(self, input_size, output_size, hyper_params):
        actor = NormalDistMLP(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hyper_params['actor_hidden_sizes'],
            mu_activation=nn.Tanh(),
            std_ones=True,
        ).to(self.device)

        critic = MLP(
            input_size=input_size,
            output_size=1,
            hidden_sizes=hyper_params['critic_hidden_sizes'],
        ).to(self.device)

        actor_optim = optim.Adam(actor.parameters(), hyper_params['actor_lr'])
        critic_optim = optim.Adam(critic.parameters(), hyper_params['critic_lr'])

        return {"actor": actor, "critic": critic, \
                "actor_optim": actor_optim, "critic_optim": critic_optim}

    def init_agent(self, env, models, device, hyper_params):
        return PPO(env, models, device, hyper_params)