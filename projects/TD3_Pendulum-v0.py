import gym
import numpy as np
import torch.nn as nn
import torch.optim as optim

from common.envs.core import GymEnv
from common.abstract.base_project import BaseProject
from algorithms.utils.update import hard_update

from common.models.mlp import MLP
from algorithms.TD3 import TD3

class Project(BaseProject):
    def init_hyper_params(self):
        return {
            "gamma": 0.99,
            "tau": 0.01,
            "noise_type": 'gaussian',
            "noise_std": np.random.uniform(0.0, 1),
            "noise_clip": np.random.uniform(0.0, 1),
            "noise_decay_period": 100000,
            "policy_update_period": 2,
            "batch_size": np.random.randint(4, 128),
            "memory_size": 1000000,            
            "max_episode_steps": 0,
            "actor_lr": np.random.uniform(0.0001, 0.01),
            "critic_lr": np.random.uniform(0.0001, 0.01),
            "actor_hidden_sizes": np.random.randint(3) * [np.random.randint(4, 256)],
            "critic_hidden_sizes": np.random.randint(3) * [np.random.randint(4, 256)],
        }

    def init_env(self, hyper_params, render_on, monitor_func):
        return GymEnv(
            env_id = 'Pendulum-v0', 
            n_envs = 1,
            render_on = render_on,
            max_episode = 500,
            max_episode_steps = hyper_params['max_episode_steps'],
            video_call_func = monitor_func(lambda x: x % 50 == 0)
        )

    def init_models(self, input_size, output_size, hyper_params):
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
        return TD3(env, models, device, hyper_params)
