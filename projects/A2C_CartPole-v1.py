import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from common.envs.core import GymEnv
from common.abstract.base_project import BaseProject
from common.models.mlp import SepActorCritic
from algorithms.A2C import A2C

class Project(BaseProject):
    def init_hyper_params(self):
        return {
            "gamma": 0.99,
            "entropy_ratio": 0.001,
            "rollout_len": 5,
            
            "n_workers": 4,
            "max_episode_steps": 0,
            "learning_rate": 0.005,
            "actor_hidden_sizes": [24],
            "critic_hidden_sizes": [24],
        }

    def init_env(self, hyper_params, render_on, monitor_func):
        return GymEnv(
            env_id = 'CartPole-v1', 
            n_envs = hyper_params['n_workers'],
            render_on = render_on,
            max_episode = 300,
            max_episode_steps = hyper_params['max_episode_steps'],
            monitor_func = monitor_func(lambda x: x % 50 == 0)
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

        optimizer = optim.Adam(model.parameters(), hyper_params['learning_rate'])

        return {'model': model, 'optim': optimizer}

    def init_agent(self, env, model, device, hyper_params):
        return A2C(env, model['model'], model['optim'], device, hyper_params)