import torch.nn as nn
import torch.optim as optim

from environments.gym import Gym
from common.abstract.base_project import BaseProject
from algorithms.models.mlp import MLP
from algorithms.DQN import DQN

class Project(BaseProject):
    def init_hyperparams(self):
        return {
            "eps_start": 1.0,
            "eps_end": 0.1,
            "eps_decay_steps": 5000,            
            "target_update_period": 600,
            "memory_size": 20000,
            "start_learning_step": 1000,
            "batch_size": 64,
            "discount_factor": 0.99,
            
            "n_worker": 8,
            "max_episode_steps": 0,

            "learning_rate": 0.001,            
            "hidden_size": [256, 256],      
        }

    def init_env(self, hyperparams):
        return Gym(
            env_id='CartPole-v1',
            n_envs=hyperparams['n_worker'],
            is_render=self.is_render,
            max_episode=500,
            max_episode_steps=hyperparams['max_episode_steps'],
            monitor_func=self.monitor_func(lambda x: x % 50 == 0),
            recent_score_len=20,
        )

    def init_model(self, env, hyperparams):
        def modeling():
            return MLP(
                input_size=env.state_size,
                output_size=env.action_size,
                hidden_sizes=hyperparams['hidden_size'],
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
