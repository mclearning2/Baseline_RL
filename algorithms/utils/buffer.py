import numpy as np

class ReplayMemory():
    '''Experience Replay Memory with numpy'''
    def __init__(self, capacity: int):
        ''' 
        Class variables
         
            _max_size(int) : The maximum size of Replay Memory
            _index(int) : The current index of memory
            _cur_size(int) : The current size of stacked memory 
            _memory(list) : The list of numpy array of data        

        '''
        self._max_size = capacity
        self._index = 0
        self._cur_size = 0

        self._memory:list = list()

    def __len__(self):
        return self._cur_size
      
    def save(self, *args):
        ''' 여러 개를 동시에 저장할 수 있다. 
            Multiprocessing에 의해 여러 개가 동시에 들어오더라도 그에 맞춰
            반복(for)해서 넣어준다.
            반드시 shape는 [n_worker, *data_shape] 여야 한다.

        Examples:
            memory = ReplayMemory(5)
            a = np.random.random([3,4])
            memory.save(a)
            print(memory._memory)

        Result:
            [array([[0.72206104, 0.90690707, 0.89969195, 0.62111934],
                    [0.39235712, 0.98103127, 0.44090238, 0.38650715],
                    [0.12436865, 0.03454921, 0.63520709, 0.68090937],
                    [0.        , 0.        , 0.        , 0.        ],
                    [0.        , 0.        , 0.        , 0.        ]])]
        '''
        # n_worker : the number of multiprocessing
        n_worker = np.shape(args[0])[0]

        for arg in args:
            assert np.shape(arg)[0] == n_worker, \
                   f"save할 데이터의 n_worker는 {n_worker}여야 합니다." \
                   f"하지만 {np.shape(arg)[0]} 입니다."
            assert np.ndim(arg) > 1, "shape는 [n_worker, *data_shape] 여야 합니다."

        # 메모리 리스트가 비어있을 경우(초기)
        # 해당 데이터 저장을 위한 array 생성 후 리스트에 추가
        if self._cur_size == 0:            
            for arg in args:
                arg_shape = np.shape(arg)[1:]
                arg_memory_size = [self._max_size] + list(arg_shape)

                self._memory.append(np.zeros(arg_memory_size, dtype=arg.dtype))
        
        for arg_index, arg in enumerate(args):
            self._memory[arg_index][self._index]
            for n in range(n_worker):
                mem_index = (self._index + n) % self._max_size
                self._memory[arg_index][mem_index] = arg[n]

        self._cur_size = min(self._cur_size + n_worker, self._max_size)
        self._index = (self._index + n_worker) % self._max_size

  
    def sample(self, n_sample):
        rand_indexes = np.random.randint(0, self._cur_size, n_sample)

        sampled_memory = list()
        for values in self._memory:
            sampled_memory.append(values[rand_indexes])

        return sampled_memory

class PrioritizedReplayMemory(ReplayMemory):
    def __init__(self, capacity, prob_alpha=0.6):
        super().__init__(capacity)

        self._prob_alpha = prob_alpha
        self._priorities = np.ones([capacity], dtype=np.float32)
    
    def save(self, *args):
        n_worker = np.shape(args[0])[0]
        super().save(*args)

        for i in range(n_worker):
            
            max_prio = self._priorities.max()
            self._priorities[self._index - n_worker + i] = max_prio

    def sample(self, batch_size, beta=0.4):
        prios = self._priorities[:self._cur_size]
        probs = prios ** self._prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(self._cur_size, batch_size, p=probs)

        weights = (self._cur_size * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        sampled_memory = list()
        for values in self._memory:
            sampled_memory.append(values[indices])
        
        sampled_memory.append(indices)
        sampled_memory.append(weights)

        return sampled_memory

    def update_priorities(self, indices, priorities):
        self._priorities[indices] = priorities