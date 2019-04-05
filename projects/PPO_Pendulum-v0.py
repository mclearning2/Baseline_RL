import gym
import multiprocessing
import numpy as np
import torch.nn as nn
import torch.optim as optim

from environments.multiprocessing_env import make_sync_env
from models.mlp import NormalDistMLP, MLP
from common.abstract.base_project import BaseProject

from algorithms.PPO import PPO

class Project(BaseProject):
    
    def init_hyper_params(self):

        return {
            "gamma": 0.99,
            "tau": 0.97,
            "epsilon": 0.2,
            "entropy_rate": 0.001,
            "rollout_len": 78,
            "mini_batch_size": 98,
            "epoch": 17,
            "n_workers": 4,
            "max_episode_steps": 400,
            "actor_lr": 0.00086,
            "critic_lr": 0.007993,
            "actor_hidden_sizes": [113, 113],
            "critic_hidden_sizes": [118, 118],
        }
    
    def init_env(self, hyper_params):
        env_id = 'Pendulum-v0'
        if self.test_mode:
            env = gym.make(env_id)
        else:
            #TODO: max_step에서 done이 True가 되는데 이를 포착할 방법을 알아내기
            env = make_sync_env(
                env_id, 
                n_envs=hyper_params['n_workers'],
                max_episode_steps=hyper_params['max_episode_steps'],
                video_call_func=self.monitor_func(lambda x:x % 20 == 0)
            )

        return env

    def init_models(self, input_size, output_size, hyper_params):
        actor = NormalDistMLP(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hyper_params['actor_hidden_sizes'],
            mu_activation=nn.Tanh(),
            std_ones=True
        ).to(self.device)

        critic = MLP(
            input_size=input_size,
            output_size=1,
            hidden_sizes=hyper_params['critic_hidden_sizes'],
        ).to(self.device)

        actor_optim = optim.Adam(actor.parameters(), hyper_params['actor_lr'])
        critic_optim = optim.Adam(critic.parameters(), hyper_params['critic_lr'])

        return {"actor": actor, "critic": critic, \
                "actor_optim": actor_optim, "critic_optim": critic_optim}

    def init_agent(self, env, models, device, hyper_params):
        return PPO(
            env=env,
            actor=models['actor'],
            critic=models['critic'],
            actor_optim=models['actor_optim'],
            critic_optim=models['critic_optim'],
            device=device,
            max_episode_steps=hyper_params["max_episode_steps"],
            n_workers=hyper_params['n_workers'],
            gamma=hyper_params['gamma'],
            tau=hyper_params['tau'],
            epsilon=hyper_params['epsilon'],
            rollout_len=hyper_params['rollout_len'],
            entropy_rate=hyper_params['entropy_rate'],
            epoch=hyper_params['epoch'],
            mini_batch_size=hyper_params['mini_batch_size']
        )

    def train(self, agent):
        agent.train(
            n_episode=500,
            recent_score_len=30
        )

    def test(self, agent, render):
        agent.test(
            n_episode=10,
            render=render
        )