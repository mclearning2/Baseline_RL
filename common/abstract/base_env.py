# This code reference
# https://github.com/openai/baselines/tree/master/baselines/common/vec_env
# But modified a little...

import gym
import numpy as np

from typing import Callable
from collections import deque
from abc import ABC, abstractmethod

from multiprocessing import Process, Pipe, cpu_count

def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)                
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'seed':
            env.seed(data)
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            env.render()
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'random_action':
            random_action = env.action_space.sample()
            remote.send(random_action)
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class VecEnv(object):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """

    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """

    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """

    def close(self):
        """
        Clean up the environments' resources.
        """

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class MultipleEnv(VecEnv):
    def __init__(self, 
        env_id, 
        n_envs,
        max_episode_steps=None,
        monitor_func=None,
    ):
        
        self.env_id = env_id
        self.n_envs = n_envs
        self.max_episode_steps = max_episode_steps

        env_fns = self._make_env_fns(monitor_func)

        self.waiting = False
        self.closed = False        
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])

        self.ps = [
            Process(
                target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_fn))) \
                    for (work_remote, remote, env_fn) \
                    in zip(self.work_remotes, self.remotes,env_fns)
                    ]
        
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, self.n_envs, observation_space, action_space)
        
        self._set_space(observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True 

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def seed(self, seed):
        assert isinstance(seed, int)
        seed_start = seed
        seed_end = seed + len(self.remotes)
        seeds = np.arange(seed_start, seed_end).tolist()

        for remote, seed in zip(self.remotes, seeds):
            remote.send(('seed', seed))
        self.waiting = True

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))
        self.waiting = False

    def env(self):
        def close():
            self.close()

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def random_action(self):
        for remote in self.remotes:
            remote.send(('random_action', None))    
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
            self.closed = True

    def _gen_env_func(self, monitor_func=None):
        def _thunk():
            env = gym.make(self.env_id)

            if self.max_episode_steps:
                env._max_episode_steps = self.max_episode_steps
            else:
                self.max_episode_steps = env._max_episode_steps

            if monitor_func:
                env = monitor_func(env)

            return env
        return _thunk

    def _make_env_fns(self, monitor_func):
        envs = []
        for i in range(self.n_envs):
            if i == 0:
                envs.append(self._gen_env_func(monitor_func=monitor_func))
            else:
                envs.append(self._gen_env_func())
        return envs

    def _set_space(self, observation_space, action_space):
        state_size = observation_space.shape        
        if len(state_size) == 1:
            self.state_size = state_size[0]
        else:
            self.state_size = state_size
        
        self.is_discrete = isinstance(action_space, gym.spaces.Discrete)
        if self.is_discrete:
            self.low = self.high = None
            self.action_size = action_space.n
        else:
            self.low = action_space.low
            self.high = action_space.high   
            self.action_size = action_space.shape[0]

    def __len__(self):
        return self.n_envs
    

class Gym(ABC):
    def __init__(
        self,
        env_id: str,
        n_envs: int,
        render_available = False,
        max_episode: int = 50000,
        max_episode_steps: int = None,
        recent_score_len: int = 100,
        monitor_func: Callable = lambda x : x,
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
        if self.render_available:
            self.env.render()

    def seed(self, seed):
        self.env.seed(seed)
    
    def random_action(self):
        return self.env.random_action()

    def first_env_episode_done(self):
        return self.episodes[0] > self.max_episode


