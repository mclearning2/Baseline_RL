import torch.nn as nn
import torch.optim as optim

from common.envs.classic import Classic
from common.abstract.base_project import BaseProject
from common.models.mlp import CategoricalDist, MLP, SepActorCritic
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

    def init_env(self, hyper_params, render_available, monitor_func):
        return Classic(
            env_id = 'CartPole-v1', 
            n_envs = hyper_params['n_workers'],
            max_episode = 300,
            max_episode_steps = hyper_params['max_episode_steps'],
            monitor_func = monitor_func(lambda x: x % 50 == 0),
            recent_score_len=20,
        )

    def init_model(self, state_size, action_size, device, hyper_params):
        actor = CategoricalDist(
            input_size=state_size,
            hidden_sizes=hyper_params['actor_hidden_sizes'],
            output_size=action_size
        )

        critic = MLP(
            input_size=state_size,
            hidden_sizes=hyper_params['critic_hidden_sizes'],
            output_size=1
        )

        model = SepActorCritic(actor, critic).to(device)

        optimizer = optim.Adam(model.parameters(), hyper_params['learning_rate'])

        return {'model': model, 'optim': optimizer}

    def init_agent(self, env, model, device, hyper_params, tensorboard_path):
        return A2C(
            env = env, 
            model = model['model'], 
            optim = model['optim'], 
            device = device, 
            hyper_params = hyper_params,
            tensorboard_path = tensorboard_path)