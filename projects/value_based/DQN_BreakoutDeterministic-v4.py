import torch.nn as nn
import torch.optim as optim

from environments.atari import Atari
from common.abstract.base_project import BaseProject
from algorithms.models.cnn import CNN
from algorithms.DQN import DQN

class Project(BaseProject):
    def init_hyperparams(self):
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

    def init_env(self, hyperparams):
        return Atari(
            env_id = 'BreakoutDeterministic-v4',
            max_episode = 10000,
            is_render=self.i_render,
            max_episode_steps = hyperparams['max_episode_steps'],
            monitor_func = self.monitor_func(lambda x: x % 50 == 0 and x > 50000),
            recent_score_len = 100,
            n_history= hyperparams['n_history'],
            width=hyperparams['img_width'],
            height=hyperparams['img_height']
        )

    def init_model(self, env, hyperparams):

        def modeling():
            return CNN(
                input_size=env.state_size,
                output_size=env.action_size,
                conv_layers= [
                    nn.Conv2d(env.state_size[0], 32, 8, 4),
                    nn.Conv2d(32, 64, 4, 2),
                    nn.Conv2d(64, 64, 3, 1),
                ],
                hidden_sizes= [512],
                hidden_activation=nn.ReLU(),
            ).to(self.device)

        online_net = modeling()
        target_net = modeling()
        target_net.eval()

        optimizer = optim.Adam(online_net.parameters(), hyperparams['learning_rate'], eps=0.01)
        
        return {"online_net": online_net, "target_net": target_net,
                "optim": optimizer}
    
    def init_agent(self, env, model, hyperparams):

        return DQN(
            env=env, 
            online_net=model['online_net'],
            target_net=model['target_net'],
            optim=model['optim'],
            device=self.device,
            hyperparams=hyperparams,
            tensorboard_path=self.tensorboard_path
        )
