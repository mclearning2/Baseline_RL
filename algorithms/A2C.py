import gym
import hues
import wandb
import torch
import numpy as np
from collections import deque
from typing import List, Union
from common.abstract.base_agent import BaseAgent
from torch.distributions import Normal, Categorical

class A2C(BaseAgent):
    def __init__(
        self,
        env: Union[List[gym.Env], gym.Env],
        actor: torch.nn.Module, 
        critic: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_optim: torch.optim.Optimizer,
        device: str,
        gamma: float = 0.99,
        entropy_rate: float = 0.001,
        rollout_len: int = 5,
        ):

        ''' A2C: Synchronous Advantage Actor Critic (called in Open AI)

        Args:
            env (list[gym.Env] or gym.Env): 
                - synchronous environments (train)
                - single environment (test)
            actor (torch.nn.Module): actor model
            critic (torch.nn.Module): critic model
            optimizer (torch.optim.Optimizer): optimizer for model
            device (str): "cuda:0" or "cpu"
            gamma (float): discount factor
            entropy_rate (float): entropy rate to multiple and add to loss
            rollout_len (int): update period by step
        '''
        self.env = env

        self.device = device
        self.actor, self.critic = actor, critic
        self.actor_optim, self.critic_optim = actor_optim, critic_optim

        self.gamma = gamma
        self.ent_rate = entropy_rate
        self.rollout_len = rollout_len

        self.values, self.log_probs, self.rewards, self.masks = [], [], [], []
        self.entropy = 0

    def select_action(self, dist: Union[Normal, Categorical]):
        state = torch.FloatTensor(dist).to(self.device)
        
        dist = self.actor.forward(state)
        value = self.critic.forward(state)
        
        action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        
        return action.cpu().numpy(), log_prob, value, entropy

    def append_memory(self, value, log_prob, reward, done, entropy):
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
        self.masks.append(torch.FloatTensor(1-done).unsqueeze(1).to(self.device))
        self.entropy += entropy      

    def compute_return(self, last_value):
        R = last_value
        returns = []
        for step in reversed(range(len(self.rewards))):
            R = self.rewards[step] + self.gamma * R * self.masks[step]
            returns.insert(0, R)
            
        return returns
    
    def train_model(self, last_state):
        last_state = torch.FloatTensor(last_state).to(self.device)
        last_value = self.critic(last_state)

        returns = self.compute_return(last_value)
        
        returns = torch.cat(returns)
        log_probs = torch.cat(self.log_probs)
        values = torch.cat(self.values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean() \
                    + self.ent_rate * self.entropy
        critic_loss = advantage.pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.values, self.log_probs, self.rewards, self.masks = [], [], [], []
        self.entropy = 0

    def train(self, **kargs):
        score = 0
        episode = 0
        state = self.env.reset()
        recent_scores = deque(maxlen=kargs['recent_score_len'])
        while episode < kargs['n_episode']:
            
            self.entropy = 0
            for _ in range(self.rollout_len):

                action, log_prob, value, entropy = self.select_action(state)

                next_state, reward, done, _ = self.env.step(action)

                self.append_memory(value, log_prob, reward, done, entropy)

                state = next_state
                episode += int(done[0])
                score += reward[0]

                if done[0]:
                    recent_scores.append(score)
                    self.write_log(
                        episode=episode,
                        score=score,
                        recent_scores=np.mean(recent_scores)
                    )                    
                    score = 0
            
            self.train_model(state)

    def test(self, **kargs):
        for episode in range(kargs['n_episode']):
            state = self.env.reset()

            score = 0
            done = False
            while not done:
                action, _, _, _ = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                score += reward
                state = next_state

                if kargs['render']:
                    self.env.render()
            
            print(f"score : {score}")

        
            