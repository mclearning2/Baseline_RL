import gym
import roboschool
import numpy as np
import multiprocessing
from copy import deepcopy
from typing import Callable
from collections import deque

from common.envs.multiprocessing_env import SubprocVecEnv

class GymEnv:
    def __init__(
        self,
        env_id: str, 
        n_envs: int = multiprocessing.cpu_count(), 
        render_on: bool = False,
        max_episode: int = 1000,
        max_episode_steps: int = 0,
        max_step_not_done: bool = True,
        last_step_reward: int = 0,
        monitor_func: Callable = None,
        recent_score_len: int = 100,
        action_scale: bool = True,
        action_clip: bool = True,
    ):
        # 정보를 얻기 위한 환경
        env = gym.make(env_id)

        self.name = env_id
        self.n_envs = n_envs
        self.render_on = render_on
        self.max_episode = max_episode
        self.max_episode_steps = max_episode_steps
        self.max_step_not_done = max_step_not_done
        self.last_step_reward = last_step_reward
        self.action_scale = action_scale
        self.action_clip = action_clip
        
        self.prev_done = np.zeros(n_envs)
        self.steps = np.zeros(n_envs, dtype=int)
        self.episodes = np.zeros(n_envs, dtype=int)
        self.scores = np.zeros(n_envs)
        self.recent_scores: deque = deque(maxlen=recent_score_len)

        state_size = env.observation_space.shape
        if len(state_size) == 1:
            self.state_size = env.observation_space.shape[0]
        else:
            self.state_size = env.observation_space.shape

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_size = env.action_space.n
            self.is_discrete = True
            self.low = None
            self.high = None
        else:
            self.action_size = env.action_space.shape[0]
            self.is_discrete = False
            self.low = env.action_space.low
            self.high = env.action_space.high
        
        envs = []
        for i in range(n_envs):
            if i == 0:
                envs.append(self.gen_env(monitor_func=monitor_func))
            else:
                envs.append(self.gen_env())
        
        self.envs = SubprocVecEnv(envs)
        

    def reset(self):
        ''' reset은 처음에 한 번만 하고 그 이후 하지 않는다.
            multiprocessing_env 내에서 알아서 done이 되면 reset 한다.        
        '''
        return self.envs.reset()

    def step(self, action: np.ndarray):
        self.render()

        # 이전의 done에 의해 리셋
        # done 하자마자 0값으로 바꿀 수 없기 때문에 이후 step에서 리셋함
        self.steps[np.where(self.prev_done)] = 0
        self.scores[np.where(self.prev_done)] = 0
        self.episodes[np.where(self.prev_done)] += 1

        self.steps += 1

        if not self.is_discrete:
            if self.action_scale:
                scale = (self.high - self.low) / 2
                reloc = self.high - scale
                action = action * scale + reloc

            if self.action_clip:
                action = np.clip(action, self.low, self.high)
        try:
            next_state, reward, done, info = self.envs.step(action)
        except EOFError:
            if self.is_discrete:
                raise TypeError("You must use Categorical distribution.") 
            else:
                raise TypeError("You must use Normal distribution.") 

        self.prev_done = done
        
        reward[np.where(self.steps == self.max_episode_steps)] += self.last_step_reward

        if self.max_step_not_done:
            done[np.where(self.steps == self.max_episode_steps)] = False
        
        self.scores += reward
        if done[0]:
            self.recent_scores.append(self.scores[0])

        return next_state, reward, done, info

    def close(self):
        self.envs.close()

    def render(self):
        if self.render_on:
            self.envs.render()

    def seed(self, seed):
        self.envs.seed(seed)
        
    def gen_env(self, monitor_func=None):
        def _thunk():
            env = gym.make(self.name)

            if self.max_episode_steps > 0:
                env._max_episode_steps = self.max_episode_steps
            else:
                self.max_episode_steps = env._max_episode_steps
            
            if monitor_func:
                env = monitor_func(env)

            return env
        return _thunk