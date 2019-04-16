import numpy as np

class ReplayMemory():
    def __init__(self, capacity: int):
        self._max_size = capacity
        self._index = 0
        self._cur_size = 0

        self._memory = list()

    def __len__(self):
        return self._cur_size
      
    def save(self, *args):
        if self._cur_size == 0:
            for value in args:
                value_size = list(np.shape(value))
                self._memory.append(np.zeros([self._max_size] + value_size))

        for index, value in enumerate(args):
            self._memory[index][self._index] = value

        if self._cur_size < self._max_size:
            self._cur_size += 1

        self._index = (self._index + 1) % self._max_size
  
    def sample(self, n_sample):
        rand_indexes = np.random.randint(0, self._cur_size, n_sample)

        sampled_memory = list()
        for values in self._memory:
            sampled_memory.append(values[rand_indexes])

        return sampled_memory