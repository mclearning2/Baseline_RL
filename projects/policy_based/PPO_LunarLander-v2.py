import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from gym.gym import Gym
from common.abstract.base_project import BaseProject
from algorithms.models.mlp import CategoricalDist, MLP, SepActorCritic
from algorithms.PPO import PPO

class Project(BaseProject):
    def init_hyperparams(self):

        return {
            "gamma": 0.99,
            "tau": 0.95,
            "epsilon": 0.3844,
            "entropy_ratio": 0.001,
            "rollout_len": 18,
            "batch_size": 55,
            "epoch": 7,
            "n_workers": 10,
            "max_episode_steps": 0,
            "actor_lr": 0.001751,
            "critic_lr": 0.0001991,
            "actor_hidden_sizes": [],
            "critic_hidden_sizes": [102,102],
        }
    
    def init_env(self, hyperparams, monitor_func):
        return Gym(
            env_id='LunarLander-v2', 
            n_envs=hyperparams['n_workers'],
            max_episode= 500,
            max_episode_steps=hyperparams['max_episode_steps'],
            monitor_func=monitor_func(lambda x: x % 50 == 0),
        )

    def init_model(self, input_size, output_size, device, hyperparams):
        actor = CategoricalDist(
            input_size=input_size,
            hidden_sizes=hyperparams['actor_hidden_sizes'],
            output_size=output_size,
            output_activation=nn.Softmax(),
        )

        critic = MLP(
            input_size=input_size,
            hidden_sizes=hyperparams['critic_hidden_sizes'],
            output_size=1
        )

        model = SepActorCritic(actor, critic).to(device)

        optimizer = optim.Adam(model.parameters(), hyperparams['actor_lr'])

        return {"model": model, "optim": optimizer}

    def init_agent(self, env, models, device, hyperparams):
        return PPO(env, models['model'], models['optim'], device, hyperparams)