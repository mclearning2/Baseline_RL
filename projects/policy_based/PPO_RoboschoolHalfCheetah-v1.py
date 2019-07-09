import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from environments.gym import Gym
from common.abstract.base_project import BaseProject
from algorithms.models.mlp import NormalDist, MLP, SepActorCritic
from algorithms.PPO import PPO

class Project(BaseProject):
    def init_hyperparams(self):
        import numpy as np
        gamma = np.random.uniform(0.9, 0.99)
        return {
            "gamma": gamma,
            "tau": np.random.uniform(0.9, gamma),
            "epsilon": np.random.uniform(0.2, 0.4),
            "entropy_ratio": 0.001,
            "rollout_len": np.random.randint(4, 128),
            "batch_size": np.random.randint(4, 128),
            "epoch": np.random.randint(4, 128),
            
            "n_workers": np.random.randint(1, 16 + 1),
            "max_episode_steps": 0,
            "actor_lr": np.random.uniform(0.00001, 0.01),
            "critic_lr": np.random.uniform(0.00001, 0.01),
            "actor_hidden_sizes": np.random.randint(4) * [np.random.randint(4, 128)],
            "critic_hidden_sizes": np.random.randint(4) * [np.random.randint(4, 128)],
        }
    
    def init_env(self, hyperparams):
        return Gym(
            env_id='RoboschoolHalfCheetah-v1', 
            n_envs=hyperparams['n_workers'],
            max_episode= 2000,
            max_episode_steps=hyperparams['max_episode_steps'],
            monitor_func=self.monitor_func(lambda x: x % 50 == 0),
            scale_action=True,
        )

    def init_model(self, env, hyperparams):

        actor = NormalDist(
            input_size=env.state_size,
            hidden_sizes=hyperparams['actor_hidden_sizes'],
            output_size=env.action_size,
            output_activation=nn.Tanh()
        )

        critic = MLP(
            input_size=env.state_size,
            hidden_sizes=hyperparams['critic_hidden_sizes'],
            output_size=1
        )

        model = SepActorCritic(actor, critic).to(self.device)

        optimizer = optim.Adam(model.parameters(), hyperparams['actor_lr'])

        return {"model": model, "optim": optimizer}

    def init_agent(self, env, models, hyperparams):
        return PPO(env, models['model'], models['optim'], \
                   self.device, hyperparams, self.tensorboard_path)