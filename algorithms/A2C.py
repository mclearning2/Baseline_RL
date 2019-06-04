import numpy as np
import torch
import torch.nn as nn
from typing import List

from common.abstract.base_agent import BaseAgent

class A2C(BaseAgent):
    ''' Synchronous Advantage Actor Critic

    - Continuous, Discrete environments are available
    - A model must output [distribution, value]
    
    Hyperparameters:
        gamma(float): discount factor
        entropy_ratio(float): 탐험을 위해 entropy을 얼마나 쓸지 계수
        rollout_len(int): 업데이트 주기 n-step
    '''
    def __init__(self, env, model, optim, device, hyper_params, tensorboard_path):

        super().__init__(tensorboard_path)

        self.env = env
        self.device = device
        self.model = model
        self.optim = optim

        self.hp = hyper_params

        self.memory_reset()

    def memory_reset(self):
        self.values: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.entropy = 0

    def select_action(self, state: np.ndarray) -> np.ndarray:
    
        state = torch.FloatTensor(state).to(self.device)
        dist, value = self.model.forward(state)
        
        action = dist.sample()

        self.values.append(value)
        self.log_probs.append(dist.log_prob(action))
        self.entropy += dist.entropy().mean()

        return action.cpu().numpy()

    def compute_return(self, last_value: torch.Tensor) -> List[torch.Tensor]:
        gamma = self.hp['gamma']

        R = last_value # []
        returns: List[torch.Tensor] = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + gamma * R * self.masks[step]
            returns.insert(0, R)
            
        return returns
    
    def train_model(self, last_state: torch.Tensor):
        last_state = torch.FloatTensor(last_state).to(self.device)
        _, last_value = self.model(last_state)

        returns = self.compute_return(last_value)
        
        returns = torch.cat(returns)
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss \
             - self.hp['entropy_ratio'] * self.entropy.mean()
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def train(self):
        state = self.env.reset()

        while not self.env.first_env_episode_done():
            for _ in range(self.hp['rollout_len']):

                action = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                reward = torch.FloatTensor(reward).to(self.device)
                done = torch.FloatTensor(done.astype(np.float)).to(self.device)
                self.rewards.append(reward)
                self.masks.append(1-done)
                                
                state = next_state

                if done[0]:
                    self.write_log(
                        global_step=self.env.episodes[0],
                        episode= self.env.episodes[0],
                        score=self.env.scores[0],
                        steps=self.env.step_per_ep[0],
                        recent_scores=np.mean(self.env.recent_scores)
                    )

            self.train_model(state)
            self.memory_reset()

        
            