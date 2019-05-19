import torch.nn as nn
import torch.optim as optim

from algorithms.utils.update import hard_update
from common.envs.gym import Gym
from common.abstract.base_project import BaseProject
from common.models.mlp import MLP
from algorithms.DDPG import DDPG

class Project(BaseProject):
    def init_hyper_params(self):
        
        import numpy as np

        return {
            "gamma": 0.99,
            "tau": 0.95,
            "batch_size": 58,
            "memory_size": 1000000,
            "max_episode_steps": 0,
            "actor_lr": 0.001165,
            "critic_lr": 0.004501,
            "actor_hidden_sizes": [57,57],
            "critic_hidden_sizes": [8,8,8],
        }

    def init_env(self, hyper_params, monitor_func):
        return Gym(
            env_id = 'Pendulum-v0', 
            n_envs = 1,
            max_episode = 500,
            max_episode_steps = hyper_params['max_episode_steps'],
            monitor_func = monitor_func(lambda x: x % 50 == 0),
            recent_score_len=20,
            scale_action=True
        )

    def init_model(self, input_size, output_size, device, hyper_params):

        actor = MLP(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hyper_params['actor_hidden_sizes'],
            output_activation=nn.Tanh()
        ).to(device)        

        target_actor = MLP(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hyper_params['actor_hidden_sizes'],
            output_activation=nn.Tanh()
        ).to(device)

        critic = MLP(
            input_size=input_size + output_size,
            output_size=1,
            hidden_sizes=hyper_params['critic_hidden_sizes'],
        ).to(device)        

        target_critic = MLP(
            input_size=input_size + output_size,
            output_size=1,
            hidden_sizes=hyper_params['critic_hidden_sizes'],
        ).to(device)

        actor_optim = optim.Adam(actor.parameters(), hyper_params['actor_lr'])
        critic_optim = optim.Adam(critic.parameters(), hyper_params['critic_lr'])

        hard_update(actor, target_actor)
        hard_update(critic, target_critic)

        return {"actor": actor, "critic": critic, \
                "target_actor": target_actor, "target_critic": target_critic, \
                "actor_optim": actor_optim, "critic_optim": critic_optim}
    
    
    def init_agent(self, env, model, device, hyper_params):
        return DDPG(
            env=env, 
            actor=model['actor'],  
            critic=model['critic'],
            target_actor=model['target_actor'], 
            target_critic=model['target_critic'],
            actor_optim=model['actor_optim'],
            critic_optim=model['critic_optim'],
            device=device,
            hyper_params=hyper_params
        )