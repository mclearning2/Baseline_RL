import gym
import torch.optim as optim

from common.utils.help_config import before_config, after_config
from common.networks.mlp import DistributedMLP
from algorithms.continuous.REINFORCE import Agent

if __name__ == '__main__':
    project_name = "MountainCarContinuous-v0"

    env = gym.make("MountainCarContinuous-v0")

    device, mode = before_config(env, device='cpu')

    hyper_params = {
        "gamma": 0.99,
        "entropy_rate": 0.001,
        "action_min": float(env.action_space.low[0]),
        "action_max": float(env.action_space.high[0]),
        "lr": 0.001,
        "mu_hidden_sizes": [],
        "sigma_hidden_sizes": []
    }

    model = DistributedMLP(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.shape[0],
        mu_hidden_sizes=hyper_params['mu_hidden_sizes'],
        sigma_hidden_sizes=hyper_params['sigma_hidden_sizes']
    ).to(device)

    optim = optim.Adam(model.parameters(), lr=hyper_params['lr'])

    agent = Agent(env, model, optim, device, hyper_params)

    if mode == 'train':
        config = {
            'n_episode': 500,
            'score_deque_len': 100,
            'save_file_name': project_name,
            'video_callable': lambda x: x > 450 and x % 10 == 0,
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
