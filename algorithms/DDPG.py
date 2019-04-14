import gym
import torch
import numpy as np
from typing import List, Union, Dict
from common.abstract.base_agent import BaseAgent

from algorithms.utils.update import soft_update
from algorithms.utils.buffer import ReplayMemory
from algorithms.utils.noise import OUNoise, GaussianNoise

class DDPG(BaseAgent):
    ''' env must be 1 '''
    def __init__(
        self,
        env,
        models: Dict[str, Union[torch.nn.Module, torch.optim.Optimizer]],
        device: str,
        hyper_params: dict
    ):
        self.env = env
        self.device = device
        self.actor, self.critic = models['actor'], models['critic']
        self.target_actor = models['target_actor'] 
        self.target_critic = models['target_critic']
        self.actor_optim = models['actor_optim']
        self.critic_optim = models['critic_optim']

        # Hyperparameters
        self.hp = hyper_params

        self.memory = ReplayMemory(self.hp['memory_size'])

        noise_type = self.hp['noise_type']
        decay_period = self.hp['noise_decay_period']

        if noise_type == 'ou':
            self.noise = OUNoise(self.env.action_size, decay_period=decay_period)
        elif noise_type == 'gaussian':
            self.noise = GaussianNoise(self.env.action_size, decay_period=decay_period)
        else:
            raise TypeError(f"noise type must be 'ou' or 'gaussian', but {noise_type}")
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(state).to(self.device)

        action = self.actor(state)

        return action.detach().cpu().numpy()

    def train_model(self):
        batch_size = self.hp['batch_size']
        tau = self.hp['tau']
        gamma = self.hp['gamma']

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Actor Loss
        state_action = torch.cat((states, self.actor(states)), -1)
        actor_loss = -self.critic(state_action).mean()
        
        # Critic Loss
        next_state_action  = torch.cat((next_states, self.target_actor(next_states) ), -1)
        target_value = self.target_critic(next_state_action)
        
        target = rewards + (1.0 - dones) * gamma * target_value

        state_action = torch.cat((states, actions), -1)
        value = self.critic(state_action)
        
        critic_loss = (value - target).pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        soft_update(self.actor, self.target_actor, tau)
        soft_update(self.critic, self.target_critic, tau)

    def train(self):
        total_step = 0
        state = self.env.reset()
        self.noise.reset()
        
        while self.env.episodes[0] < self.env.max_episode:
            total_step += 1
            
            action = self.select_action(state)
            action = self.noise.add(action, total_step)

            next_state, reward, done, info = self.env.step(action, scale=True)

            self.memory.save(state[0], action[0], reward, next_state[0], done)

            state = next_state

            if len(self.memory) > self.hp['batch_size']:
                self.train_model()

            if done[0]:
                self.noise.reset()

                self.write_log(
                    episode=self.env.episodes[0],
                    score=self.env.scores[0],
                    steps=self.env.steps[0],
                    recent_scores=np.mean(self.env.recent_scores)
                )


