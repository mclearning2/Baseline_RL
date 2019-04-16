import gym
import torch
import numpy as np
from typing import List, Union, Dict
from common.abstract.base_agent import BaseAgent

from algorithms.utils.update import soft_update
from algorithms.utils.buffer import ReplayMemory
from algorithms.utils.noise import GaussianNoise

class TD3(BaseAgent):
    ''' Deep Deterministic Policy Gradient
    
    - The only continuous environment is available
    - The input of critic is the output of actor. So critic can't be CNN
    - The action noise is used Gaussian distribution
    - Train model [actor, critic1, critic2],
      Target model [target_actor, target_critic1, target_critic2]
      Otimizer [actor_optim, critic_optim1, critic_optim2]
    - hyper_params in this agent
        gamma(float): discount_factor
        tau(float): The ratio of target model for soft update
        max_sigma(float): The start of standard deviation of gaussian noise
        min_sigma(float): The end of standard deviation of gaussian noise
        noise_decay_period(int): The number of steps to decay sigma of noise
        noise_std(float): The standard deviation of the target action noise (mean is 0)
        noise_clip(float): The clip of noise of target action (-noise_clip, +noise_clip)
        policy_update_period(int): When critic updaten this times, actor updates once
        batch_size(int): The sample size of experience replay memory
        memory_size(int): The size of experience replay memory
    '''
    def __init__(
        self,
        env,
        actor, critic1, critic2,
        target_actor, target_critic1, target_critic2,
        actor_optim, critic_optim1, critic_optim2,
        device: str,
        hyper_params: dict
    ):
        self.env = env
        self.device = device
        
        self.actor = actor
        self.critic1 = critic1
        self.critic2 = critic2

        self.target_actor = target_actor
        self.target_critic1 = target_critic1
        self.target_critic2 = target_critic2

        self.actor_optim = actor_optim
        self.critic_optim1 = critic_optim1
        self.critic_optim2 = critic_optim2

        self.hp = hyper_params

        self.memory = ReplayMemory(self.hp['memory_size'])

        max_sigma = self.hp['max_sigma']
        min_sigma = self.hp['min_sigma']
        decay_period = self.hp['noise_decay_period']
    
        self.noise = GaussianNoise(
            self.env.action_size, 
            max_sigma=max_sigma,
            min_sigma=min_sigma,
            decay_period=decay_period
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(state).to(self.device)

        action = self.actor(state)

        return action.detach().cpu().numpy()

    def train_model(self, step):
        batch_size = self.hp['batch_size']
        tau = self.hp['tau']
        gamma = self.hp['gamma']
        noise_std = self.hp['noise_std']
        noise_clip = self.hp['noise_clip']
        policy_update_period = self.hp['policy_update_period']

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Critic Loss
        next_action = self.target_actor(next_states)
        noise = torch.normal(torch.zeros_like(next_action), noise_std).to(self.device)
        noise = torch.clamp(noise, -noise_clip, noise_clip)
        next_action += noise
        
        next_state_action = torch.cat((next_states, next_action), -1)
        target_q1 = self.target_critic1(next_state_action)
        target_q2 = self.target_critic2(next_state_action)
        small_target_q = torch.min(target_q1, target_q2)
        target = rewards + (1.0 - dones) * gamma * small_target_q

        q1 = self.critic1(torch.cat((states, actions), -1))
        q2 = self.critic2(torch.cat((states, actions), -1))

        critic_loss1 = (q1 - target.detach()).pow(2).mean()
        critic_loss2 = (q2 - target.detach()).pow(2).mean()

        self.critic_optim1.zero_grad()
        critic_loss1.backward()
        self.critic_optim1.step()

        self.critic_optim2.zero_grad()
        critic_loss2.backward()
        self.critic_optim2.step()

        if step % policy_update_period == 0:
            state_action = torch.cat((states, self.actor(states)), -1)
            actor_loss = -self.critic1(state_action).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            soft_update(self.actor, self.target_actor, tau)
            soft_update(self.critic1, self.target_critic1, tau)
            soft_update(self.critic2, self.target_critic2, tau)

    def train(self):
        total_step = 0
        state = self.env.reset()
        
        while self.env.episodes[0] < self.env.max_episode + 1:
            total_step += 1
            
            action = self.select_action(state)
            action += self.noise.sample(total_step)            

            next_state, reward, done, info = self.env.step(action)
            
            self.memory.save(state[0], action[0], reward, next_state[0], done)

            state = next_state

            if len(self.memory) > self.hp['batch_size']:
                self.train_model(self.env.steps[0])

            if done[0]:
                self.write_log(
                    episode=self.env.episodes[0],
                    score=self.env.scores[0],
                    steps=self.env.steps[0],
                    recent_scores=np.mean(self.env.recent_scores)
                )