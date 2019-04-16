import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from common.envs.core import GymEnv
from common.abstract.base_project import BaseProject

from common.models.mlp import SepActorCritic
from algorithms.PPO import PPO

class Project(BaseProject):
    def init_hyper_params(self):
        import numpy as np

        return {
            "gamma": 0.99,
            "tau": 0.95,
            "epsilon": np.random.uniform(0.2, 0.4),
            "entropy_ratio": 0.001,
            "rollout_len": np.random.randint(4, 30),
            "batch_size": np.random.randint(4, 128),
            "epoch": np.random.randint(4, 128),
            "n_workers": np.random.randint(4, 12),
            "max_episode_steps": 0,
            "actor_lr": np.random.uniform(0.0001, 0.01),
            "critic_lr": np.random.uniform(0.0001, 0.01),
            "actor_hidden_sizes": np.random.randint(4) * [np.random.randint(4, 128)],
            "critic_hidden_sizes": np.random.randint(4) * [np.random.randint(4, 128)],
        }
    
    def init_env(self, hyper_params, render_on, monitor_func):
        return GymEnv(
            env_id='Acrobot-v1', 
            n_envs=hyper_params['n_workers'],
            render_on=render_on,
            max_episode= 300,
            max_episode_steps=hyper_params['max_episode_steps'],
            monitor_func=monitor_func(lambda x: x % 10 == 0),
        )

    def init_model(self, input_size, output_size, device, hyper_params):

        model = SepActorCritic(
            input_size=input_size,
            actor_output_size=output_size,
            critic_output_size=1,
            actor_hidden_sizes=hyper_params['actor_hidden_sizes'],
            critic_hidden_sizes=hyper_params['critic_hidden_sizes'],
            actor_output_activation=nn.Softmax(),
            dist=Categorical,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), hyper_params['actor_lr'])

        return {"model": model, "optim": optimizer}

    def init_agent(self, env, models, device, hyper_params):
        return PPO(env, models['model'], models['optim'], device, hyper_params)