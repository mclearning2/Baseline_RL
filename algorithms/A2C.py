import torch
import numpy as np
import torch.nn as nn
from typing import List

from common.abstract.base_agent import BaseAgent
from environments.gym import Gym
 
class A2C(BaseAgent):
    ''' Synchronous Advantage Actor Critic

    - continuous, discrete action space
    - A model must output [distribution, value]
    
    Hyperparameters:
        gamma(float): discount factor
        entropy_ratio(float): 탐험을 위해 entropy을 얼마나 쓸지 계수
        rollout_len(int): 업데이트 주기 n-step
    '''
    def __init__(self, 
        env: Gym,
        model: nn.Module,
        optim: torch.optim, 
        device: str, 
        hyperparams: dict, 
        tensorboard_path: str
    ):

        super().__init__(tensorboard_path)

        self.env = env
        self.device = device
        
        self.model = model
        self.optim = optim

        self.hp = hyperparams

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
        R = last_value
        returns = []
        for step in reversed(range(len(self.rewards))): # [rollout_len, n_worker, 1]
            R = self.rewards[step] + self.hp['gamma'] * R * self.masks[step]
            returns.insert(0, R)
        
        return returns
    
    def train_model(self, last_state: torch.Tensor):
        last_state = torch.FloatTensor(last_state).to(self.device)
        _, last_value = self.model(last_state)

        # returns shape [rollout_len] with tensor [n_worker, 1]
        returns = self.compute_return(last_value)

        returns = torch.cat(returns).squeeze()
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values).squeeze()

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

        
            