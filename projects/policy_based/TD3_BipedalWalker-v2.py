import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim

from gym.gym import Gym
from common.abstract.base_project import BaseProject
from algorithms.utils.update import hard_update

from algorithms.models.mlp import MLP
from algorithms.TD3 import TD3

class Project(BaseProject):
    def init_hyper_params(self):
        return {
            "gamma": 0.99,
            "tau": 0.995,
            "min_sigma": 0.1,
            "max_sigma": 0.1,
            "noise_decay_period": 0,
            "noise_std": 0.2,
            "noise_clip": 0.5,            
            "policy_update_period": 2,
            "batch_size": 100,
            "memory_size": 1000000,            
            "max_episode_steps": 2000,
            "actor_lr": 0.001,
            "critic_lr": 0.001,
            "actor_hidden_sizes": [400, 300],
            "critic_hidden_sizes": [400, 300]
        }

    def init_env(self, hyper_params, monitor_func):
        return Gym(
            env_id = 'BipedalWalker-v2', 
            n_envs = 1,
            max_episode = 1000,
            max_episode_steps = hyper_params['max_episode_steps'],
            monitor_func = monitor_func(lambda x: x % 50 == 0),
            scale_action = True,
        )

    def init_model(self, input_size, output_size, device, hyper_params):
        actor = MLP(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hyper_params['actor_hidden_sizes'],
            output_activation=nn.Tanh()
        ).to(self.device)

        target_actor = MLP(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hyper_params['actor_hidden_sizes'],
            output_activation=nn.Tanh()
        ).to(self.device)

        critic1 = MLP(
            input_size=input_size + output_size,
            output_size=1,
            hidden_sizes=hyper_params['critic_hidden_sizes'],
        ).to(self.device)

        critic2 = MLP(
            input_size=input_size + output_size,
            output_size=1,
            hidden_sizes=hyper_params['critic_hidden_sizes'],
        ).to(self.device)

        target_critic1 = MLP(
            input_size=input_size + output_size,
            output_size=1,
            hidden_sizes=hyper_params['critic_hidden_sizes'],
        ).to(self.device)

        target_critic2 = MLP(
            input_size=input_size + output_size,
            output_size=1,
            hidden_sizes=hyper_params['critic_hidden_sizes'],
        ).to(self.device)

        actor_optim = optim.Adam(actor.parameters(), hyper_params['actor_lr'])
        critic_optim1 = optim.Adam(critic1.parameters(), hyper_params['critic_lr'])
        critic_optim2 = optim.Adam(critic2.parameters(), hyper_params['critic_lr'])

        hard_update(actor, target_actor)
        hard_update(critic1, target_critic1)
        hard_update(critic2, target_critic2)

        return {"actor": actor, "critic1": critic1, "critic2": critic2, \
                "target_actor": target_actor, \
                "target_critic1": target_critic1, "target_critic2": target_critic2, \
                "actor_optim": actor_optim, \
                "critic_optim1": critic_optim1, "critic_optim2": critic_optim2}
    
    def init_agent(self, env, models, device, hyper_params):
        return TD3(
            env=env, 
            actor=models['actor'], 
            critic1=models['critic1'], 
            critic2=models['critic2'],
            target_actor=models['target_actor'], 
            target_critic1=models['target_critic1'], 
            target_critic2=models['target_critic2'],
            actor_optim=models['actor_optim'], 
            critic_optim1=models['critic_optim1'], 
            critic_optim2=models['critic_optim2'],
            device=device, 
            hyper_params=hyper_params
        )
