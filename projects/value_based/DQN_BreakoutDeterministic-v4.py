import torch.nn as nn
import torch.optim as optim

from gym.atari import Atari
from common.abstract.base_project import BaseProject
from algorithms.models.cnn import CNN
from algorithms.DQN import DQN

class Project(BaseProject):
    def init_hyper_params(self):
        return {
            "eps_start": 1.0,
            "eps_end": 0.1,
            "eps_decay_steps": 1000000,
            "target_update_period": 10000,
            "memory_size": 400000,
            "start_learning_step": 50000,
            "batch_size": 32,
            "discount_factor": 0.99,
            "learning_rate": 0.00025,
            "max_episode_steps": 0,
            "n_history": 4,
            "img_width": 80,
            "img_height": 80
        }

    def init_env(self, hyper_params, monitor_func):
        return Atari(
            env_id = 'BreakoutDeterministic-v4',
            max_episode = 10000,
            max_episode_steps = hyper_params['max_episode_steps'],
            monitor_func = monitor_func(lambda x: x % 50 == 0 and x > 50000),
            recent_score_len = 100,
            n_history= hyper_params['n_history'],
            width=hyper_params['img_width'],
            height=hyper_params['img_height']
        )

    def init_model(self, input_size, output_size, device, hyper_params):

        def modeling():
            return CNN(
                input_size=input_size,
                output_size=output_size,
                conv_layers= [
                    nn.Conv2d(input_size[0], 32, 8, 4),
                    nn.Conv2d(32, 64, 4, 2),
                    nn.Conv2d(64, 64, 3, 1),
                ],
                hidden_sizes= [512],
                hidden_activation=nn.ReLU(),
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
