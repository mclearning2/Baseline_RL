import gym
import torch
import numpy as np
from typing import List, Union, Dict
from common.abstract.base_agent import BaseAgent

class PPO(BaseAgent):
    def __init__(
        self,
        env: Union[List[gym.Env], gym.Env],
        models: Dict[str, Union[torch.nn.Module, torch.optim.Optimizer]],
        device: str,
        hyper_params: dict
    ):
        self.env = env
        self.device = device
        self.actor, self.critic = models['actor'], models['critic']
        self.actor_optim = models['actor_optim']
        self.critic_optim = models['critic_optim']

        # Hyperparameters
        self.hp = hyper_params

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
        gamma, tau = self.hp['gamma'], self.hp['tau']

        values = self.values + [last_value]
        gae = 0
        returns = list()

        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * self.masks[step] \
                  - values[step]
            gae = delta + gamma * tau * self.masks[step] * gae
            returns.insert(0, gae + values[step])

        return returns

    def ppo_iter(self, states, actions, log_probs, returns, advantage):
        # This batch size is different with self.batch_size
        # Because states is stacked by multiprocessing environment
        # So states.size(0) = rollout_len * n_workers
        batch_size = states.size(0)
        mini_batch_size = self.hp['mini_batch_size']

        for _ in range(batch_size // mini_batch_size):
            random_indices = np.random.randint(0, batch_size, mini_batch_size)
            yield states[random_indices, :], \
                  actions[random_indices, :], \
                  log_probs[random_indices, :], \
                  returns[random_indices, :], \
                  advantage[random_indices, :]
    
    def ppo_update(self, states, actions, log_probs, returns, advantage):
        epsilon = self.hp['epsilon']
        entropy_ratio = self.hp['entropy_ratio']
        epoch = self.hp['epoch']

        for _ in range(epoch):
            for state, action, old_log_probs, return_, adv in \
                self.ppo_iter(states, actions, log_probs, returns, advantage):

                # Actor Loss
                # ============================================================
                dist = self.actor(state)
                
                new_log_probs = dist.log_prob(action)
                ratio = (new_log_probs - old_log_probs).exp()                
                
                surr_loss = ratio * adv
                clipped_surr_loss = torch.clamp(ratio, 1.0-epsilon, 1.0+epsilon) * adv

                entropy = dist.entropy().mean()                

                actor_loss = - torch.min(surr_loss, clipped_surr_loss).mean() \
                             - entropy_ratio * entropy
                
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

    def train(self):
        state = self.env.reset()

        while self.env.episodes[0] < self.env.max_episode:
            for _ in range(self.hp['rollout_len']):
                action = self.select_action(state)
                
                
                next_state, reward, done, info = self.env.step(action)

                reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
                done = torch.FloatTensor(done.astype(np.float)).unsqueeze(1).to(self.device)
                self.rewards.append(reward)
                self.masks.append(1-done)  

                state = next_state

                if done[0]:
                    self.write_log(
                        episode=self.env.episodes[0],
                        score=self.env.scores[0],
                        steps=self.env.steps[0],
                        recent_scores=np.mean(self.env.recent_scores)
                    )

            self.train_model(state)