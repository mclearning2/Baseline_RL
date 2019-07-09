import torch
import numpy as np
from typing import List, Union, Dict
from common.abstract.base_agent import BaseAgent

from algorithms.utils.update import soft_update, hard_update
from algorithms.utils.buffer import ReplayMemory
from algorithms.utils.noise import OUNoise

class DDPG(BaseAgent):
    ''' Deep Deterministic Policy Gradient
    
    - Only continuous action space
    - Action noise : Ornstein–Uhlenbeck process

    - hyperparams in this agent
        gamma(float): discount_factor
        tau(float): The ratio of target model for soft update
        batch_size(int): The sample size of experience replay memory
        memory_size(int): The size of experience replay memory
    '''
    def __init__(
        self,
        env,
        actor, critic,
        target_actor, target_critic,
        actor_optim, critic_optim,
        device: str,
        hyperparams: dict,
        tensorboard_path: str
    ):
        super().__init__(tensorboard_path)

        self.env = env
        self.device = device

        self.actor = actor
        self.critic = critic
        self.target_actor = target_actor
        self.target_critic = target_critic

        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        
        self.hp = hyperparams

        self.memory = ReplayMemory(self.hp['memory_size'])
        self.noise = OUNoise(self.env.action_size)        

        hard_update(actor, target_actor)
        hard_update(critic, target_critic)
        
    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(state).to(self.device)

        action = self.actor(state)

        return action.detach().cpu().numpy()

    def train_model(self):
        batch_size = self.hp['batch_size']
        tau = self.hp['tau']
        gamma = self.hp['gamma']

        states, actions, rewards, next_states, dones = \
            self.memory.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones.astype(np.float)).to(self.device)

        # Actor Loss
        state_action = torch.cat((states, self.actor(states)), -1)

        actor_loss = - self.critic(state_action).mean()
        
        # Critic Loss
        next_state_action  = torch.cat((next_states, self.target_actor(next_states)), -1)
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
        state = self.env.reset()
        self.noise.reset()
        
        while not self.env.is_first_env_done():
            action = self.select_action(state)
            action += self.noise.sample()

            next_state, reward, done, info = self.env.step(action)

            self.memory.save(state, action, reward, next_state, done)

            state = next_state

            if len(self.memory) > self.hp['batch_size']:
                self.train_model()

            if done[0]:
                self.noise.reset()

                self.write_log(
                    global_step = self.env.total_step,
                    episode=self.env.episodes[0],
                    score=self.env.scores[0],
                    steps=self.env.step_per_ep[0],
                    recent_scores=np.mean(self.env.recent_scores)
                )


