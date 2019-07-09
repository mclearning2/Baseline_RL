import torch.nn as nn
import torch.optim as optim

from environments.gym import Gym
from common.abstract.base_project import BaseProject
from algorithms.models.mlp import CategoricalDist, MLP, SepActorCritic
from algorithms.A2C import A2C

class Project(BaseProject):
    def init_hyperparams(self):
        return {
            "gamma": 0.99,
            "entropy_ratio": 0.001,
            "rollout_len": 5,
            
            "n_workers": 8,
            "max_episode_steps": 0,

            "learning_rate": 0.005,
            "actor_hidden_sizes": [24],
            "critic_hidden_sizes": [24],
        }

    def init_env(self, hyperparams):
        return Gym(
            env_id = 'CartPole-v1', 
            n_envs = hyperparams['n_workers'],
            max_episode = 100,
            max_episode_steps = hyperparams['max_episode_steps'],
            monitor_func = self.monitor_func(lambda x: x % 50 == 0),
            recent_score_len=20,
            is_render=self.is_render
        )

    def init_model(self, env, hyperparams):
        actor = CategoricalDist(
            input_size=env.state_size,
            hidden_sizes=hyperparams['actor_hidden_sizes'],
            output_size=env.action_size
        )

        critic = MLP(
            input_size=env.state_size,
            hidden_sizes=hyperparams['critic_hidden_sizes'],
            output_size=1
        )

        model = SepActorCritic(actor, critic).to(self.device)

        optimizer = optim.Adam(model.parameters(), \
                               hyperparams['learning_rate'])

        return {'model': model, 'optim': optimizer}

    def init_agent(self, env, model, hyperparams):
        return A2C(
            env = env, 
            model = model['model'], 
            optim = model['optim'], 
            device = self.device, 
            hyperparams = hyperparams,
            tensorboard_path = self.tensorboard_path)
