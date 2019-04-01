import gym
import torch.nn as nn
import torch.optim as optim

from common.utils.help_config import before_config, after_config
from common.networks.mlp import MLP

from algorithms.discrete.REINFORCE import Agent

if __name__ == '__main__':
    project_name = "REINFORCE_CartPole-v1"

    env = gym.make("CartPole-v1")

    device, mode = before_config(env, device='cpu')

    hyper_params = {
        "gamma": 0.99,
        "lr": 0.04,
        "hidden_sizes": []
    }

    model = MLP(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.n,
        hidden_sizes=hyper_params['hidden_sizes'],
        output_activation=nn.Softmax()
    ).to(device)

    optim = optim.Adam(model.parameters(), hyper_params['lr'])

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
            'wandb_run_path': "mclearning2/reinforce-cartpole-v1/runs/"
                              "hf56ef3f",
            'load_file_name': project_name,
        }

    after_config(agent, model, hyper_params, mode, config)
