import cv2
import numpy as np
from typing import Callable

from common.envs.gym import Gym

def cvt_gray_resize_half(frame, height = 84, width = 84):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (width, height), 
                    interpolation=cv2.INTER_AREA)
    
    return frame

class Atari(Gym):
    def __init__(
        self,
        env_id: str, 
        max_episode: int = 10000,
        max_episode_steps: int = None,
        recent_score_len: int = 100,
        monitor_func: Callable = lambda x: x,
        n_history: int = 4,
        width: int = 84,
        height: int = 84,
        no_op: int = 30,
    ):
        super().__init__(
            env_id=env_id, 
            n_envs=1, 
            max_episode=max_episode,
            max_episode_steps=max_episode_steps,
            recent_score_len=recent_score_len,
            monitor_func=monitor_func
            )

        self.state_size = (n_history, height, width)
        self.n_history = n_history
        self.width = width
        self.height = height
        self.no_op = no_op

    def _reset(self, state):
        # 0 ~ 30 time step 동안 에이전트를 가만히 두어 랜덤하게 시작 (No-op)
        for _ in range(1, np.random.randint(self.no_op)):
            state, _, _, info = self.env.step(self.env.random_action())
            self.lives = info[0]['ale.lives']

        frame = cvt_gray_resize_half(state[0], self.height, self.width)
        self.history = np.stack([frame] * self.n_history)

    def reset(self):
        state = super().reset()
        self._reset(state)

        return np.expand_dims(self.history, axis=0)

    def step(self, action):
        next_state, reward, done, info = super().step(action)

        if done[0]:
            self._reset(next_state)
        else:
            next_state = cvt_gray_resize_half(next_state[0], self.height, self.width)
            next_state = np.expand_dims(next_state, axis=0)
            
            self.history = np.concatenate((self.history[1:, :, :], next_state), axis=0)            

        if self.lives > info[0]['ale.lives']:
            done = np.array([True])
            self.lives = info[0]['ale.lives']
        else:
            done = np.array([False])

        reward = np.clip(reward, -1, 1)

        return np.expand_dims(self.history, axis=0), reward, done, info

        
