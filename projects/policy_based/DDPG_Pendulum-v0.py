import torch.nn as nn
import torch.optim as optim

from algorithms.utils.update import hard_update
from environments.gym import Gym
from common.abstract.base_project import BaseProject
from algorithms.models.mlp import MLP
from algorithms.DDPG import DDPG

class Project(BaseProject):
    def init_hyperparams(self):
        
        return {
            "gamma": 0.99,
            "tau": 0.95,
            "batch_size": 58,
            "memory_size": 1000000,
            "max_episode_steps": 0,
            "actor_lr": 0.001165,
            "critic_lr": 0.004501,
            "actor_hidden_sizes": [57,57],
            "critic_hidden_sizes": [8,8,8],
        }

    def init_env(self, hyperparams):
        return Gym(
            env_id = 'Pendulum-v0', 
            n_envs = 1,
            max_episode = 500,
            max_episode_steps = hyperparams['max_episode_steps'],
            monitor_func = self.monitor_func(lambda x: x % 50 == 0),
            recent_score_len=20,
            scale_action=True,
            render_available=self.is_render
        )

    def init_model(self, env, hyperparams):

        actor = MLP(
            input_size=env.state_size,
            output_size=env.action_size,
            hidden_sizes=hyperparams['actor_hidden_sizes'],
            output_activation=nn.Tanh()
        ).to(self.device)        

        target_actor = MLP(
            input_size=env.state_size,
            output_size=env.action_size,
            hidden_sizes=hyperparams['actor_hidden_sizes'],
            output_activation=nn.Tanh()
        ).to(self.device)

        critic = MLP(
            input_size=env.state_size + env.action_size,
            output_size=1,
            hidden_sizes=hyperparams['critic_hidden_sizes'],
        ).to(self.device)        

        target_critic = MLP(
            input_size=env.state_size + env.action_size,
            output_size=1,
            hidden_sizes=hyperparams['critic_hidden_sizes'],
        ).to(self.device)

        actor_optim = optim.Adam(actor.parameters(), hyperparams['actor_lr'])
        critic_optim = optim.Adam(critic.parameters(), hyperparams['critic_lr'])


        return {"actor": actor, "critic": critic, \
                "target_actor": target_actor, "target_critic": target_critic, \
                "actor_optim": actor_optim, "critic_optim": critic_optim}
    
    
    def init_agent(self, env, model, device, hyperparams, tensorboard_path):

        return DDPG(
            env=env, 
            actor=model['actor'],  
            critic=model['critic'],
            target_actor=model['target_actor'], 
            target_critic=model['target_critic'],
            actor_optim=model['actor_optim'],
            critic_optim=model['critic_optim'],
            device=device,
            hyperparams=hyperparams,
            tensorboard_path = tensorboard_path
        )
