import numpy as np
from collections import deque
from common.abstract.base_env import MultipleEnv

class Gym:
    def __init__(
        self,
        env_id: str,
        n_envs: int,
        is_render = False,
        max_episode: int = 50000,
        max_episode_steps: int = None,
        recent_score_len: int = 100,
        monitor_func=lambda x : x,
        clip_action: bool = True,
        scale_action: bool = False,
    ):
        ''' OpenAI gym에서 사용할 

        Args:
            env_id(str): OpenAI gym의 환경 id
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

        self.is_render = is_render

        self.done = np.zeros(n_envs, dtype=bool)
        self.step_per_ep = np.zeros(n_envs, dtype=int)
        self.episodes = np.zeros(n_envs, dtype=int)
        self.scores = np.zeros(n_envs, dtype=float)
        self.recent_scores: deque = deque(maxlen=recent_score_len)

    def reset(self):
        ''' 초기에만 reset하고 그 이후 env 내부에서 done이 되었을 경우
            자동으로 done이 된다.
        '''
        return self.env.reset()

    def close(self):
        self.env.close()

    def step(self, action: np.ndarray):
        self.render()

        if self.is_discrete:
            assert np.shape(action) == (self.n_envs,)
        else:
            assert np.shape(action) == (self.n_envs, self.action_size)

        # step 이후에 reset하면 log를 기록할 수 없으므로 다음 step에서 reset
        self.step_per_ep[np.where(self.done)] = 0
        self.scores[np.where(self.done)] = 0
        self.episodes[np.where(self.done)] += 1

        self.step_per_ep += 1
        if not self.is_discrete:
            if self.scale_action:
                scale_factor = (self.high - self.low) / 2
                reloc_factor = self.high - scale_factor

                action = action * scale_factor + reloc_factor

            if self.clip_action:
                action = np.clip(action, self.low, self.high)
        
        next_state, reward, done, info = self.env.step(action)

        # 마지막 step까지 간 경우 done으로 할지 말지
        #done[np.where(self.step_per_ep == self.max_episode_steps)] = False

        self.scores += reward
        self.done = done

        if done[0]:
            self.recent_scores.append(self.scores[0])

        done = np.expand_dims(done, axis=1)
        reward = np.expand_dims(reward, axis=1)

        if isinstance(self.state_size, (list, tuple)):
            assert np.shape(next_state) == (self.n_envs, *self.state_size)
        else:
            assert np.shape(next_state) == (self.n_envs, self.state_size)
        assert np.shape(done) == (self.n_envs, 1)
        assert np.shape(reward) == (self.n_envs, 1)

        return next_state, reward, done, info
    
    def render(self):
        if self.is_render:
            self.env.render()

    def seed(self, seed):
        self.env.seed(seed)
    
    def random_action(self):
        return self.env.random_action()

    def first_env_ep_done(self):
        return self.episodes[0] > self.max_episode