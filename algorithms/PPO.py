import gym
import torch
import numpy as np
from copy import deepcopy
from collections import deque
from typing import List, Union
from common.abstract.base_agent import BaseAgent

class PPO(BaseAgent):
    def __init__(
        self,
        env: Union[List[gym.Env], gym.Env],
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        actor_optim: torch.optim.Optimizer,
        critic_optim: torch.optim.Optimizer,
        device: str,
        max_episode_steps: int,
        n_workers: int,
        gamma: float,
        tau: float,
        epsilon: float,
        entropy_rate: float,
        rollout_len: int,
        epoch: int,
        mini_batch_size: int
    ):
        self.env = env

        self.device = device
        self.actor, self.critic = actor, critic
        self.actor_optim, self.critic_optim = actor_optim, critic_optim

        # Hyperparameters
        self.max_episode_steps = max_episode_steps
        self.n_workers = n_workers
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.ent_rate = entropy_rate
        self.rollout_len = rollout_len
        self.epoch = epoch
        self.mini_batch_size = mini_batch_size

        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.values: list = []
        self.masks: list = []
        self.log_probs: list = []

    def select_action(self, state: np.ndarray) -> np.ndarray:

        state = torch.FloatTensor(state).to(self.device)
        
        dist = self.actor.forward(state)
        value = self.critic.forward(state)
        
        action = dist.sample()

        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.log_probs.append(dist.log_prob(action))
        
        return action.cpu().numpy()

    def compute_gae(self, last_value):
        values = self.values + [last_value]
        gae = 0
        returns = list()

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] \
                  + self.gamma * values[step + 1] * self.masks[step] \
                  - values[step]
            gae = delta + self.gamma * self.tau * self.masks[step] * gae
            returns.insert(0, gae + values[step])

        return returns

    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        # This batch size is different with self.batch_size
        # Because states is stacked by multiprocessing environment
        # So states.size(0) = rollout_len * n_workers
        batch_size = states.size(0)

        for _ in range(batch_size // self.mini_batch_size):
            random_indices = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[random_indices, :], \
                  actions[random_indices, :], \
                  log_probs[random_indices, :], \
                  returns[random_indices, :], \
                  advantage[random_indices, :]
    
    def ppo_update(self, states, actions, log_probs, returns, advantage):
        for _ in range(self.epoch):
            for state, action, old_log_probs, return_, adv in \
                self.ppo_iter(states, actions, log_probs, returns, advantage):

                # Actor Loss
                # ============================================================
                dist = self.actor(state)
                
                new_log_probs = dist.log_prob(action)
                ratio = (new_log_probs - old_log_probs).exp()                
                
                surr_loss = ratio * adv
                clipped_surr_loss = torch.clamp(
                    ratio, 1.0-self.epsilon, 1.0+self.epsilon
                ) * adv

                entropy = dist.entropy().mean()                

                actor_loss = - torch.min(surr_loss, clipped_surr_loss).mean() \
                             - self.ent_rate * entropy
                
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                # ============================================================

                # Critic Loss
                #TODO: clip value by epsilon, clip gradient by norm
                # ============================================================
                value = self.critic(state)
                
                critic_loss = 0.5 * (return_ - value).pow(2).mean()
                
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
                # ============================================================

    def train_model(self, last_state):
        last_state = torch.FloatTensor(last_state).to(self.device)
        last_value = self.critic(last_state)

        returns = self.compute_gae(last_value)

        returns = torch.cat(returns).detach() 
        log_probs = torch.cat(self.log_probs).detach() 
        values = torch.cat(self.values).detach()
        states = torch.cat(self.states) 
        actions = torch.cat(self.actions)
        
        advantage = returns - values 

        self.ppo_update(states, actions, log_probs, returns, advantage)

        self.values, self.log_probs, self.rewards, self.masks = [], [], [], []
        self.states, self.actions = [], []

    def train(self, recent_score_len, n_episode):
        step = np.zeros(self.n_workers, dtype=np.int)
        episode = 0
        state = self.env.reset()
        recent_scores = deque(maxlen=recent_score_len)

        while episode < n_episode:
            score = 0
            
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
