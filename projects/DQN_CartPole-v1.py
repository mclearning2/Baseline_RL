import torch.nn as nn
import torch.optim as optim

from common.envs.gym import Gym
from common.abstract.base_project import BaseProject
from common.models.mlp import MLP
from algorithms.DQN import DQN

class Project(BaseProject):
    def init_hyper_params(self):
        return {
            "eps_start": 1.0,
            "eps_end": 0.01,
            "eps_decay_steps": 10000,
            "target_update_period": 300,
            "memory_size": 2000,
            "start_learning_step": 1000,
            "batch_size": 64,
            "discount_factor": 0.99,
            "learning_rate": 0.001,
            "max_episode_steps": 500,
            "hidden_size": [10],
        }

    def init_env(self, hyper_params, monitor_func):
        return Gym(
            env_id='CartPole-v1',
            n_envs=8,
            max_episode=3000,
            max_episode_steps=hyper_params['max_episode_steps'],
            monitor_func=monitor_func(None),#monitor_func(lambda x: x % 50 == 0),
            recent_score_len=20,
        )

    def init_model(self, input_size, output_size, device, hyper_params):
        def modeling():
            return MLP(
                input_size=input_size,
                output_size=output_size,
                hidden_sizes=hyper_params['hidden_size'],
                output_activation=nn.Softmax(-1)
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
