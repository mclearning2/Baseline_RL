import gym
import torch
import numpy as np
from copy import deepcopy
from collections import deque
from typing import List, Union, Tuple
from common.abstract.base_agent import BaseAgent
from torch.distributions import Normal

class A2C(BaseAgent):
    ''' A2C: Synchronous Advantage Actor Critic (called in Open AI) '''
    def __init__(
        self,
        env: Union[List[gym.Env], gym.Env],
        actor: torch.nn.Module, 
        critic: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_optim: torch.optim.Optimizer,
        device: str,
        n_workers: int,
        max_episode_steps: int,
        gamma: float = 0.99,
        entropy_rate: float = 0.001,
        rollout_len: int = 5,
        ):

        ''' Initialize

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

        # Hyperparameters
        self.max_episode_steps = max_episode_steps
        self.n_workers = n_workers
        self.gamma = gamma
        self.ent_rate = entropy_rate
        self.rollout_len = rollout_len

        self.values: list = []
        self.log_probs: list = []
        self.rewards: list = []
        self.masks: list = []
        self.entropy = 0

    def select_action(self, state: np.ndarray) -> np.ndarray:
        
        state = torch.FloatTensor(state).to(self.device)
        
        dist = self.actor.forward(state)
        value = self.critic.forward(state)
        
        action = dist.sample()

        self.values.append(value)
        self.log_probs.append(dist.log_prob(action))
        self.entropy += dist.entropy().mean()
        
        return action.cpu().numpy()

    def compute_return(self, last_value):
        R = last_value
        returns = list()
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
                    - self.ent_rate * self.entropy
        critic_loss = advantage.pow(2).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()
        
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        self.values: list = []
        self.log_probs: list = []
        self.rewards: list = []
        self.masks: list = []
        self.entropy = 0

    def train(self, recent_score_len, n_episode):
        step = np.zeros(self.n_workers, dtype=np.int)
        score = 0
        episode = 0
        state = self.env.reset()
        recent_scores = deque(maxlen=recent_score_len)

        while episode < n_episode:
            for _ in range(self.rollout_len):
                step += 1

                action = self.select_action(state)

                next_state, reward, done, info = self.env.step(action)

                done_bool = deepcopy(done)
                if self.max_episode_steps:
                    done_bool[np.where(step == self.max_episode_steps)] = False
                
                self.rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
                self.masks.append(torch.FloatTensor(1-done_bool).unsqueeze(1).to(self.device))
                                
                state = next_state
                score += reward[0]
                episode += done.sum()

                if done[0]:
                    print(type(episode), type(score), type(step[0]), type(recent_scores))
                    recent_scores.append(score)
                    self.write_log(
                        episode=episode,
                        score=score,
                        steps=step[0],
                        recent_scores=np.mean(recent_scores)
                    )
                    score = 0

                step[np.where(done)] = 0
            
            self.train_model(state)

    def test(self, n_episode, render):
        for ep in range(n_episode):
            state = self.env.reset()
            score = 0
            done = False
            while not done:
                if render:
                    self.env.render()
                action = select_action(state)

                next_state, reward, done, _ = self.env.step(action)

                score += reward

            print("score", score)

        
            