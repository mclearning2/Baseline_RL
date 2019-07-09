import torch.nn as nn
import torch.optim as optim

from environments.gym import Gym
from common.abstract.base_project import BaseProject
from algorithms.models.mlp import CategoricalDist, MLP, SepActorCritic
from algorithms.PPO import PPO

class Project(BaseProject):
    def init_hyperparams(self):

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

            "learning_rate": 0.007007,            
            "actor_hidden_sizes": [],
            "critic_hidden_sizes": [70],
        }
    
    def init_env(self, hyperparams):
        return Gym(
            env_id='Acrobot-v1', 
            n_envs=hyperparams['n_workers'],
            max_episode=120,
            max_episode_steps=hyperparams['max_episode_steps'],
            monitor_func=self.monitor_func(lambda x: x % 30 == 0),
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

        optimizer = optim.Adam(model.parameters(), hyperparams['learning_rate'])

        return {"model": model, "optim": optimizer}

    def init_agent(self, env, models, hyperparams):
        return PPO(env, models['model'], models['optim'], \
                   self.device, hyperparams, self.tensorboard_path)