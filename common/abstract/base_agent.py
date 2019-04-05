import wandb
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def train_model(self, *args, **kargs):
        pass

    @abstractmethod
    def train(self, **kargs):
        pass

    @abstractmethod
    def test(self, **kargs):
        pass

    def write_log(self, **kargs):
        s = ""
        for name, value in kargs.items():
            if isinstance(value, (int, np.integer)):
                s += f"{name} : {value} | "
            else:
                s += f"{name} : {value:.3f} | "
            
        print(s)
    
        wandb.log(kargs)
