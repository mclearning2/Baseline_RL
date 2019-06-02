import wandb
import numpy as np
from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def test(self):
        state = self.env.reset()
        while not self.env.is_episode_done():
            action = self.select_action(state)

            next_state, reward, done, _ = self.env.step(action)

            state = next_state

            print(reward)

    def write_log(self, **kargs):
        """ Write print and logging.

        (e.g) 
        score_value = 10.5
        step_value = 3
        episode = 5

        write_log(
            score = score_value, 
            step = step_value, 
            episode = episode_value
        )
        >>> score : 10.5 | step : 3 | episode : 5 |

        """
        s = ""
        for name, value in kargs.items():
            if isinstance(value, (int, np.integer)):
                s += f"{name} : {value} | "
            else:
                s += f"{name} : {value:.3f} | "
            
        print(s)
    
        wandb.log(kargs)
