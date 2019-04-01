import gym
import torch.nn as nn
import torch.optim as optim

from common.utils.help_config import before_config, after_config
from common.networks.mlp import MLP

from algorithms.discrete.A2C import Agent

if __name__ == '__main__':
    project_name = "A2C_CartPole-v1"

    env = gym.make("CartPole-v1")

    device, mode = before_config(env, device='cpu')

    hyper_params = {
        "gamma": 0.99,
        "entropy_rate": 0,
        "actor_lr": 0.001,
        "critic_lr": 0.005,
        "actor_hidden_sizes": [24],
        "critic_hidden_sizes": [24, 24],
    }

    actor = MLP(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        hidden_sizes=hyper_params['actor_hidden_sizes'],
        output_activation=nn.Softmax()
    ).to(device)

    critic = MLP(
        input_size=env.observation_space.shape[0],
        output_size=1,
        hidden_sizes=hyper_params['critic_hidden_sizes']
    ).to(device)

    actor_optim = optim.Adam(actor.parameters(), hyper_params['actor_lr'])
    critic_optim = optim.Adam(critic.parameters(), hyper_params['critic_lr'])

    model = (actor, critic)
    optim = (actor_optim, critic_optim)

    agent = Agent(env, model, optim, device, hyper_params)

    if mode == 'train':
        config = {
            'n_episode': 500,
            'score_deque_len': 100,
            'save_file_name': project_name,
            'video_callable': lambda x: x > 400 and x % 10 == 0,
        }
    elif mode == 'test':
        config = {
            'render': True,
            'n_episode': 5,
            'wandb_run_path': "mclearning2/a2c-cartpole-v1/runs/k1fey909",
            'load_file_name': project_name,
        }

    after_config(agent, model, hyper_params, mode, config)
