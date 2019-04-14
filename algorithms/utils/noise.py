# Reference
# - https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/gaussian_strategy.py
# - https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py

import gym
import numpy as np

class GaussianNoise(object):
    def __init__(self, action_size, max_sigma=1.0, min_sigma=1.0, decay_period=1000000):
        self.action_size = action_size
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
    
    def reset(self):
        pass

    def add(self, action, t=0):
        sigma  = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        action += np.random.randn(self.action_size) * sigma
        return action

class OUNoise(object):
    def __init__(
        self, 
        action_size,
        mu=0.0, 
        theta=0.15, 
        max_sigma=0.3, 
        min_sigma=0.3, 
        decay_period=100000
    ):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_size  = action_size
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_size) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_size)
        self.state = x + dx
        return self.state
    
    def add(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return action + ou_state