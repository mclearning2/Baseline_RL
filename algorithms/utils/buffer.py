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
            반복(for)해서 넣어준다. 하지만 n_worker가 1 이상이여야하기 때문에
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

        # if memory is empty
        if self._cur_size == 0:
            # save arguments
            for arg in args:
                # [memory size, The shape of argument without n_worker]

                arg_shape = list(np.shape(arg))[1:] 
                
                if arg_shape == []:
                    arg_shape = [1]

                arg_memory_size = [self._max_size] + arg_shape

                # Generate memory(numpy) with zeros
                self._memory.append(np.zeros(arg_memory_size, dtype=arg.dtype))

        # 
        for arg_index, arg in enumerate(args):
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