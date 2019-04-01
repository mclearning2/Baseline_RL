import gym
import torch.optim as optim

from common.utils.help_config import before_config, after_config
from common.networks.mlp import DistributedMLP, MLP

from algorithms.continuous.A2C import Agent

if __name__ == '__main__':
    project_name = "A2C_MountainCarContinuous-v0"

    env = gym.make("MountainCarContinuous-v0")

    device, mode = before_config(env, device='cpu')

    import numpy as np

    hyper_params = {
        "gamma": 0.99,
        "entropy_rate": np.random.uniform(0.001, 0.1),
        "actor_lr": np.random.uniform(0.0001, 0.01),
        "critic_lr": np.random.uniform(0.0005, 0.05),
        "actor_mu_hidden_sizes": np.random.randint(3) * [np.random.randint(10, 64)],
        "actor_sigma_hidden_sizes": np.random.randint(3) * [np.random.randint(10, 64)],
        "critic_hidden_sizes": np.random.randint(3) * [np.random.randint(10, 64)],
        "action_min": float(env.action_space.low[0]),
        "action_max": float(env.action_space.high[0]),
    }

    # test stknek ltnk ltnseknlt et lksnt klsnt lknksn tklsknt n ets nlkne
    # lknt lknek ltnlks ntlk ntlkse ltls
    actor = DistributedMLP(
        input_size=env.observation_space.shape[0],
        output_size=env.action_space.shape[0],
        mu_hidden_sizes=hyper_params['actor_mu_hidden_sizes'],
        sigma_hidden_sizes=hyper_params['actor_sigma_hidden_sizes']
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

    if mode == "train":
        config = {
            'n_episode': 50,
            'score_deque_len': 10,
            'save_file_name': project_name,
            'video_callable': lambda x: x > 45
        }

    elif mode == 'test':
        config = {
            'render': True,
            'n_episode': 5,
            'wandb_run_path': None,
            'load_file_name': project_name,
        }

    after_config(agent, model, hyper_params, mode, config)
