import gym
import numpy as np
import torch.optim as optim

from environments.multiprocessing_env import make_sync_env
from models.mlp import CategoricalMLP, MLP
from common.abstract.base_project import BaseProject

from algorithms.A2C import A2C

class Project(BaseProject):
    def init_hyper_params(self):
        return {
            "gamma": 0.99,
            "entropy_rate": 0.001,
            "rollout_len": 5,
            "n_workers": 16,
            "actor_lr": 0.001,
            "critic_lr": 0.005,
            "actor_hidden_sizes": [24],
            "critic_hidden_sizes": [24, 24],
        }

    def init_env(self, hyper_params):
        env_id = 'CartPole-v1'
        if self.config.test:
            env = gym.make(env_id)
        else:
            env = make_sync_env(
                env_id = env_id, 
                n_envs = hyper_params['n_workers'],
                video_func = self.monitor,
                video_callable = lambda x: x > 500 and x % 20 == 0)
        
        return env

    def init_models(self, env, hyper_params):
        actor = CategoricalMLP(
            input_size=env.observation_space.shape[0],
            output_size=env.action_space.n,
            hidden_sizes=hyper_params['actor_hidden_sizes'],
        ).to(self.device)

        critic = MLP(
            input_size=env.observation_space.shape[0],
            output_size=1,
            hidden_sizes=hyper_params['critic_hidden_sizes']
        ).to(self.device)

        actor_optim = optim.Adam(actor.parameters(), hyper_params['actor_lr'])
        critic_optim = optim.Adam(critic.parameters(), hyper_params['critic_lr'])

        return {"actor": actor, "critic": critic,
                "actor_optim": actor_optim, "critic_optim": critic_optim}

    def init_agent(self, env, models, device, hyper_params):

        return A2C(
            env=env,
            actor=models['actor'],
            critic=models['critic'],
            actor_optim=models['actor_optim'],
            critic_optim=models['critic_optim'],
            device=device,
            gamma=hyper_params['gamma'],
            entropy_rate=hyper_params['entropy_rate'],
        )

    def train(self, agent):
        agent.train(
            n_episode=10000,
            recent_score_len=30,
        )

    def test(self, agent, render):
        agent.test(
            n_episode=10,
            render=render
        )
