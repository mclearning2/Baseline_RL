import torch.nn as nn
import torch.optim as optim

from gym.gym import Gym
from common.abstract.base_project import BaseProject
from algorithms.models.mlp import DuelingMLP
from algorithms.DQN import DQN

class Project(BaseProject):
    def init_hyper_params(self):
        return {
            "eps_start": 1.0,
            "eps_end": 0.1,
            "eps_decay_steps": 5000,
            "n_worker": 8,
            "target_update_period": 600,
            "memory_size": 20000,
            "start_learning_step": 1000,
            "batch_size": 64,
            "discount_factor": 0.99,
            "learning_rate": 0.001,
            "max_episode_steps": 0,
            "hidden_size": [256],
            "advantage_hidden_size": [128],
            "value_hidden_size": [128],
        }

    def init_env(self, hyper_params, monitor_func):
        return Gym(
            env_id='CartPole-v1',
            n_envs=hyper_params['n_worker'],
            max_episode=500,
            max_episode_steps=hyper_params['max_episode_steps'],
            monitor_func=monitor_func(lambda x: x % 50 == 0),
            recent_score_len=20,
        )

    def init_model(self, input_size, output_size, device, hyper_params):
        def modeling():
            return  DuelingMLP(
                input_size=input_size,
                hidden_sizes=hyper_params["hidden_size"],
                output_sizes1=hyper_params['advantage_hidden_size'] + [output_size],
                output_sizes2=hyper_params['value_hidden_size'] + [1],
            ).to(device)

        online_net = modeling()
        target_net = modeling()
        target_net.eval()

        optimizer = optim.Adam(online_net.parameters(), hyper_params['learning_rate'], eps=0.01)
        
        return {"online_net": online_net, "target_net": target_net,
                "optim": optimizer}
    
    def init_agent(self, env, model, device, hyper_params):

        return DQN(
            env=env, 
            online_net=model['online_net'],
            target_net=model['target_net'],
            optim=model['optim'],
            device=device,
            hyper_params=hyper_params
        )
