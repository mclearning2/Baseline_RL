import gym
import torch
import numpy as np
from typing import List, Union, Dict
from common.abstract.base_agent import BaseAgent

from algorithms.utils.update import soft_update
from algorithms.utils.buffer import ReplayMemory
from algorithms.utils.noise import OUNoise, GaussianNoise

class TD3(BaseAgent):
    def __init__(
        self,
        env,
        models: Dict[str, Union[torch.nn.Module, torch.optim.Optimizer]],
        device: str,
        hyper_params: dict
    ):
        self.env = env
        self.device = device
        
        self.actor = models['actor']
        self.critic1 = models['critic1']
        self.critic2 = models['critic2']

        self.target_actor = models['target_actor']
        self.target_critic1 = models['target_critic1']
        self.target_critic2 = models['target_critic2']

        self.actor_optim = models['actor_optim']
        self.critic_optim1 = models['critic_optim1']
        self.critic_optim2 = models['critic_optim2']

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

    def train_model(self, step):
        batch_size = self.hp['batch_size']
        tau = self.hp['tau']
        gamma = self.hp['gamma']
        noise_std = self.hp['noise_std']
        noise_clip = self.hp['noise_clip']
        policy_update_period = self.hp['policy_update_period']

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

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
        self.noise.reset()
        
        while self.env.episodes[0] < self.env.max_episode:
            total_step += 1
            
            action = self.select_action(state)
            action = self.noise.add(action, total_step)

            next_state, reward, done, info = self.env.step(action, scale=True)

            self.memory.save(state[0], action[0], reward, next_state[0], done)

            state = next_state

            if len(self.memory) > self.hp['batch_size']:
                self.train_model(self.env.steps[0])

            if done[0]:
                self.noise.reset()

                self.write_log(
                    episode=self.env.episodes[0],
                    score=self.env.scores[0],
                    steps=self.env.steps[0],
                    recent_scores=np.mean(self.env.recent_scores)
                )