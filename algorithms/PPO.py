import gym
import torch
import numpy as np
from common.abstract.base_agent import BaseAgent

class PPO(BaseAgent):
    ''' Proximal Policy Optimization

    - Continuous, Discrete environments are available
    - A model must output [distribution, value]
    - An optim must be used
    - hyper_params in this agent
        gamma(float): Discount factor
        tau(float): Lambda in GAE
        epsilon(float): Clip bound of surrogate loss (1-epsilon, 1+epsilon)
        entropy_ratio(float): The ratio of entropy multiplied by the actor loss
        rollout_len(int): The period of interaction and update
        batch_size(int): The batch size of training
        epoch(int): The number of ppo update epoch
    '''
    def __init__(self, env, model, optim, device, hyper_params):
        self.env = env
        self.device = device
        self.model = model
        self.optim = optim

        self.hp = hyper_params
        
        self.memory_reset()

    def memory_reset(self):
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

    def select_action(self, state: np.ndarray) -> np.ndarray:

        state = torch.FloatTensor(state).to(self.device)
        
        dist, value = self.model(state)
        
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
        # states.size(0) = rollout_len * n_workers
        memory_size = states.size(0)
        batch_size = self.hp['batch_size']

        for _ in range(memory_size // batch_size):
            random_indices = np.random.randint(0, memory_size, batch_size)
            yield states[random_indices], \
                  actions[random_indices], \
                  log_probs[random_indices], \
                  returns[random_indices], \
                  advantage[random_indices]
    
    def ppo_update(self, states, actions, log_probs, returns, advantage):
        epsilon = self.hp['epsilon']
        entropy_ratio = self.hp['entropy_ratio']
        epoch = self.hp['epoch']

        for _ in range(epoch):
            for state, action, old_log_probs, return_, adv in \
                self.ppo_iter(states, actions, log_probs, returns, advantage):

                dist, value = self.model(state)

                # Actor Loss
                # ============================================================
                new_log_probs = dist.log_prob(action)
                ratio = (new_log_probs - old_log_probs).exp()                
                
                surr_loss = ratio * adv
                clipped_surr_loss = torch.clamp(ratio, 1.0-epsilon, 1.0+epsilon) * adv

                entropy = dist.entropy().mean()                

                actor_loss = - torch.min(surr_loss, clipped_surr_loss).mean() \
                             - entropy_ratio * entropy
                # ============================================================

                # Critic Loss
                #TODO: clip value by epsilon, clip gradient by norm
                # ============================================================
                critic_loss = (return_ - value).pow(2).mean()
                # ============================================================

                loss = actor_loss + 0.5 * critic_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

    def train_model(self, last_state):
        last_state = torch.FloatTensor(last_state).to(self.device)
        _, last_value = self.model(last_state)

        returns = self.compute_gae(last_value)

        returns = torch.cat(returns).detach() 
        log_probs = torch.cat(self.log_probs).detach() 
        values = torch.cat(self.values).detach()
        states = torch.cat(self.states) 
        actions = torch.cat(self.actions)
        
        advantage = returns - values 

        self.ppo_update(states, actions, log_probs, returns, advantage)

    def train(self):
        state = self.env.reset()
        while self.env.episodes[0] < self.env.max_episode + 1:
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
                        episode=self.env.episodes[0],
                        score=self.env.scores[0],
                        steps=self.env.steps[0],
                        recent_scores=np.mean(self.env.recent_scores)
                    )

            self.train_model(state)
            self.memory_reset()