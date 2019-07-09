import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim

from environments.gym import Gym
from common.abstract.base_project import BaseProject
from algorithms.utils.update import hard_update

from algorithms.models.mlp import MLP
from algorithms.TD3 import TD3

class Project(BaseProject):
    def init_hyperparams(self):
        return {
            "gamma": 0.99,
            "tau": 0.99,
            "min_sigma": 1.0,
            "max_sigma": 0.2,
            "noise_decay_period": 992171,
            "noise_std": 0.4803,
            "noise_clip": 0.4499,
            "policy_update_period": 3,
            "batch_size": 17,
            "memory_size": 624399,
            "max_episode_steps": 0,
            "actor_lr": 0.002759,
            "critic_lr": 0.002508,
            "actor_hidden_sizes": [63, 63],
            "critic_hidden_sizes": [110, 110]
        }

    def init_env(self, hyperparams):
        return Gym(
            env_id = 'Pendulum-v0', 
            n_envs = 1,
            max_episode = 500,
            max_episode_steps = hyperparams['max_episode_steps'],
            monitor_func = self.monitor_func(lambda x: x % 50 == 0),
            scale_action = True,
        )

    def init_model(self, env, hyperparams):
        actor = MLP(
            input_size=env.state_size,
            output_size=env.action_size,
            hidden_sizes=hyperparams['actor_hidden_sizes'],
            output_activation=nn.Tanh()
        ).to(self.device)

        target_actor = MLP(
            input_size=env.state_size,
            output_size=env.action_size,
            hidden_sizes=hyperparams['actor_hidden_sizes'],
            output_activation=nn.Tanh()
        ).to(self.device)

        critic1 = MLP(
            input_size=env.state_size + env.action_size,
            output_size=1,
            hidden_sizes=hyperparams['critic_hidden_sizes'],
        ).to(self.device)

        critic2 = MLP(
            input_size=env.state_size + env.action_size,
            output_size=1,
            hidden_sizes=hyperparams['critic_hidden_sizes'],
        ).to(self.device)

        target_critic1 = MLP(
            input_size=env.state_size + env.action_size,
            output_size=1,
            hidden_sizes=hyperparams['critic_hidden_sizes'],
        ).to(self.device)

        target_critic2 = MLP(
            input_size=env.state_size + env.action_size,
            output_size=1,
            hidden_sizes=hyperparams['critic_hidden_sizes'],
        ).to(self.device)

        actor_optim = optim.Adam(actor.parameters(), hyperparams['actor_lr'])
        critic_optim1 = optim.Adam(critic1.parameters(), hyperparams['critic_lr'])
        critic_optim2 = optim.Adam(critic2.parameters(), hyperparams['critic_lr'])

        hard_update(actor, target_actor)
        hard_update(critic1, target_critic1)
        hard_update(critic2, target_critic2)

        return {"actor": actor, "critic1": critic1, "critic2": critic2, \
                "target_actor": target_actor, \
                "target_critic1": target_critic1, "target_critic2": target_critic2, \
                "actor_optim": actor_optim, \
                "critic_optim1": critic_optim1, "critic_optim2": critic_optim2}
    
    def init_agent(self, env, models, hyperparams):
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
            device=self.device, 
            hyperparams=hyperparams
        )
