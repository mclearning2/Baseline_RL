import numpy as np

class ReplayMemory():
    def __init__(self, capacity: int):
        self._states = None
        self._actions = None
        self._rewards = None
        self._next_states = None
        self._dones = None

        self._max_size = capacity
        self._index = 0
        self._cur_size = 0

    def __len__(self):
        return self._cur_size
      
    def save(self, state, action, reward, next_state, done):
        state_size = list(np.shape(state))

        if self._cur_size == 0:
            self._states = np.zeros([self._max_size] + state_size)
            self._actions = np.zeros([self._max_size])
            self._rewards = np.zeros([self._max_size])
            self._next_states = np.zeros([self._max_size] + state_size)
            self._dones = np.zeros([self._max_size])

        self._states[self._index]      = state
        self._actions[self._index]     = action
        self._rewards[self._index]     = reward
        self._next_states[self._index] = next_state
        self._dones[self._index]       = done

        if self._cur_size < self._max_size:
            self._cur_size += 1

        self._index = (self._index + 1) % self._max_size
  
    def sample(self, n_sample):
        rand_indexes = np.random.randint(0, self._cur_size, n_sample)

        return self._states[rand_indexes], \
               self._actions[rand_indexes], \
               self._rewards[rand_indexes].astype(np.float), \
               self._next_states[rand_indexes], \
               self._dones[rand_indexes].astype(np.float)