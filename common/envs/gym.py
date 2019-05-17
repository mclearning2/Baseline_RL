import gym
import numpy as np

from typing import Callable
from collections import deque

from common.envs.multiprocessing_env import MultipleEnv

class Gym:
    def __init__(
        self,
        env_id: str, 
        n_envs: int,
        max_episode: int = 50000,
        max_episode_steps: int = None,
        recent_score_len: int = 100,
        monitor_func: Callable = None,
        clip_action: bool = True,
        scale_action: bool = False,
    ):
    
        self.env = MultipleEnv(env_id, n_envs, max_episode_steps, monitor_func)
        
        self.env_id = env_id
        self.n_envs = n_envs
        self.max_episode = max_episode
        self.max_episode_steps = self.env.max_episode_steps

        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.is_discrete = self.env.is_discrete
        self.low, self.high = self.env.low, self.env.high
        self.clip_action = clip_action
        self.scale_action = scale_action
        
        self.render_available = False
        
        self.done = np.zeros(n_envs, dtype=bool)
        self.steps = np.zeros(n_envs, dtype=int)
        self.episodes = np.zeros(n_envs, dtype=int)
        self.scores = np.zeros(n_envs)
        self.recent_scores: deque = deque(maxlen=recent_score_len)

    def reset(self):
        ''' reset은 처음에 한 번만 하고 그 이후 하지 않는다.
            multiprocessing_env 내에서 알아서 done이 되면 reset 한다.        
        '''
        return self.env.reset()

    def step(self, action: np.ndarray):
        self.render()

        self.steps[np.where(self.done)] = 0
        self.scores[np.where(self.done)] = 0
        self.episodes[np.where(self.done)] += 1

        self.steps += 1

        if not self.is_discrete:
            if self.scale_action:
                scale_factor = (self.high - self.low) / 2
                reloc_factor = self.high - scale_factor

                action = action * scale_factor + reloc_factor

            if self.clip_action:
                action = np.clip(action, self.low, self.high)
        
        next_state, reward, done, info = self.env.step(action)

        done[np.where(self.steps == self.max_episode_steps)] = False

        self.scores += reward
        self.done = done

        if done[0]:
            self.recent_scores.append(self.scores[0])

        done = np.expand_dims(done, axis=1)
        reward = np.expand_dims(reward, axis=1)

        return next_state, reward, done, info

    def close(self):
        self.env.close()

    def render(self):
        if self.render_available:
            self.env.render()

    def seed(self, seed):
        self.env.seed(seed)

    def random_action(self):
        return self.env.random_action()

    def is_episode_done(self):
        return self.max_episode < self.episodes[0]