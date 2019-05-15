import numpy as np

class ReplayMemory():
    def __init__(self, capacity: int):
        self._max_size = capacity
        self._index = 0
        self._cur_size = 0

        self._memory:list = list()

    def __len__(self):
        return self._cur_size
      
    def save(self, *args):
        batch_size = np.shape(args[0])[0]

        if self._cur_size == 0:
            for value in args:
                value_size = list(np.shape(value))

                self._memory.append(np.zeros([self._max_size] + value_size[1:], \
                                    dtype=value.dtype))

        for mem_i, value in enumerate(args):
            for n in range(batch_size):
                index = (self._index + n) % self._max_size
                self._memory[mem_i][index] = value[n]

        self._cur_size = min(self._cur_size + batch_size, self._max_size)
        self._index = (self._index + batch_size) % self._max_size

  
    def sample(self, n_sample):
        rand_indexes = np.random.randint(0, self._cur_size, n_sample)

        sampled_memory = list()
        for values in self._memory:
            sampled_memory.append(values[rand_indexes])

        return sampled_memory