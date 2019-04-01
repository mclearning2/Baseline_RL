import gym
import torch.nn as nn
import torch.optim as optim

from common.utils.help_config import before_config, after_config
from common.networks.mlp import MLP

from algorithms.discrete.SynchronousA2C import Agent
from environments.multiprocessing_env import SubprocVecEnv

if __name__ == '__main__':
    project_name = "SynchronousA2C_CartPole-v1"

    num_envs = 16
    env_name = "CartPole-v1"

    def make_env():
        def _thunk():
            env = gym.make(env_name)
            return env

        return _thunk

    env = gym.make(env_name)

    device, mode = before_config(env)

    if mode == 'train':
        envs = [make_env() for i in range(num_envs)]
        envs = SubprocVecEnv(envs)
    elif mode == 'test':
        envs = env

    hyper_params = {
        "gamma": 0.99,
        "n_workers": num_envs,
        "batch_size": 2,
        "entropy_rate": 0.03519,
        "actor_lr": 0.006505,
        "critic_lr": 0.003143,
        "actor_hidden_sizes": [41, 41],
        "critic_hidden_sizes": [62, 62]
    }

    actor = MLP(
        input_size=envs.observation_space.shape[0],
        output_size=envs.action_space.n,
        hidden_sizes=hyper_params['actor_hidden_sizes'],
        output_activation=nn.Softmax()
    ).to(device)

    critic = MLP(
        input_size=envs.observation_space.shape[0],
        output_size=1,
        hidden_sizes=hyper_params['critic_hidden_sizes']
    ).to(device)

    actor_optim = optim.Adam(actor.parameters(), hyper_params['actor_lr'])
    critic_optim = optim.Adam(critic.parameters(), hyper_params['critic_lr'])

    model = (actor, critic)
    optim = (actor_optim, critic_optim)

    agent = Agent(envs, model, optim, device, hyper_params)

    if mode == 'train':
        config = {
            'n_episode': 500,
            'save_file_name': project_name,
            'score_deque_len': 100
        }
    elif mode == 'test':
        config = {
            'render': True,
            'n_episode': 5,
            'wandb_run_path': None,
            'load_file_name': project_name,
        }

    after_config(agent, model, hyper_params, mode, config)
