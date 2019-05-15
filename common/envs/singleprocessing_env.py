import gym
import numpy as np

class SingleEnv():
    def __init__(self, 
        env_id, 
        max_episode_steps=None,
        monitor_func=lambda x:x
    ):
        env = gym.make(env_id)

        self.env_id = env_id    
        
        self.max_episode_steps = max_episode_steps
        if self.max_episode_steps:
            env._max_episode_steps = self.max_episode_steps
        else:
            self.max_episode_steps = env._max_episode_steps
    
        self.env = monitor_func(env)
        self._set_space(env.observation_space, env.action_space)
        
    def seed(self, seed):
        self.env.seed(seed)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def random_action(self):
        return np.array((self.env.action_space.sample(),))
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        ob, rew, done, info = self.env.step(action)

        if done:
            self.env.reset()
        return np.array([ob]), np.array([rew]), np.array([done]), (info,)

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