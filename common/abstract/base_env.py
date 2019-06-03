import gym

from typing import Callable
from collections import deque
from abc import ABC, abstractmethod

from common.envs.multiprocessing_env import MultipleEnv

class Gym:
    def __init__(
        self,
        env_id: str,
        n_envs: int,
        render_available = False,
        max_episode: int = 50000,
        max_episode_steps: int = None,
        recent_score_len: int = 100,
        monitor_func: Callable,
        clip_action: bool = True,
        scale_action: bool = False,
    ):
    ''' OpenAI gym에서 사용할 

    Args:
        env_id: OpenAI gym의 환경 id
        n_envs: 멀티프로세싱을 위한 worker 수
        max_episode: 최대 에피소드 수. 이 수를 넘으면 종료
        max_episode_steps: 최대 step 수. 이 수가 지나면 done
        recent_score_len: 한 에피소드의 보상의 합인 score(return)을 저장하고
                          평균을 구할 때 최근 몇 에피소드를 구할 지
        monitor_func: video를 record할 때 쓰는 함수
        clip_action: action을 -1과 1사이로 클립할 지 여부 (continuous만)
        scale_action: action을 환경의 low와 high에 맞게 정규화
                      (-1 ~ 1 -> low ~ high)
    '''

        self.env = MultipleEnv(env_id, n_envs, max_episode_steps, monitor_func)

        self.env_id = env_id
        self.n_envs = n_envs
        self.max_episode = max_episode
        self.max_episode_steps = max_episode_steps

        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.is_discrete = self.env.is_discrete
        self.low, self.high = self.env.low, self.env.high
        self.clip_action = clip_action
        self.scale_action = scale_action

        self.render_available = render_available

        self.done = np.zeros(n_envs, dtype=bool)
        self.step_per_ep = np.zeros(n_envs, dtype=int)
        self.episodes = np.zeros(n_envs, dtype=int)
        self.scores = np.zeros(n_envs, dtype=int)
        self.recent_scores: deque = deque(maxlen=recent_score_len)

    def reset(self):
        