import gym
import numpy as np
import torch.optim as optim

from common.networks.mlp import DistributedMLP, MLP
from common.abstract.base_project import BaseProject

from algorithms.continuous.GAE import GAE


class Project(BaseProject):
    def init_hyper_params(self):
        td_lambda = np.random.uniform(0.9, 1.0)
        actor_size = np.random.randint(1) * [np.random.randint(1,256)]
        return {
            "gamma": np.random.uniform(td_lambda, 1.0),
            "td_lambda": td_lambda,
            "entropy_rate": np.random.uniform(0.0001, 0.1),
            "actor_lr": np.random.uniform(0.00001, 0.01),
            "critic_lr": np.random.uniform(0.00001, 0.01),
            "batch_size": np.random.randint(1, 1024),  # n-step
            "actor_mu_hidden_sizes": [],
            "actor_sigma_hidden_sizes": [],
            "critic_hidden_sizes": [],
        }

    def init_env(self):
        env = gym.make('Pendulum-v0')
        video_callable = lambda x: x > 950 and x % 10 == 0
        return env, video_callable

    def init_models(self, env, hyper_params):
        actor = DistributedMLP(
            input_size=env.observation_space.shape[0],
            output_size=env.action_space.shape[0],
            mu_hidden_sizes=hyper_params['actor_mu_hidden_sizes'],
            sigma_hidden_sizes=hyper_params['actor_sigma_hidden_sizes'],
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

        return GAE(
            env=env,
            actor=models['actor'],
            critic=models['critic'],
            actor_optim=models['actor_optim'],
            critic_optim=models['critic_optim'],
            device=device,
            gamma=hyper_params['gamma'],
            td_lambda=hyper_params['td_lambda'],
            entropy_rate=hyper_params['entropy_rate'],
            batch_size=hyper_params['batch_size'],
            action_space_low=env.action_space.low[0],
            action_space_high=env.action_space.high[0]
        )

    def train(self, agent, render):
        agent.train(
            n_episode=1000,
            render=render,
            score_deque_len=30,
        )

    def test(self, agent, render):
        agent.test(
            n_episode=10,
            render=render
        )
