import wandb
import numpy as np
from abc import ABC, abstractmethod

from torch.utils.tensorboard import SummaryWriter

class BaseAgent(ABC):
    def __init__(self, tensorboard_path):
        self.writer = SummaryWriter(tensorboard_path)        

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
        while not self.env.first_env_episode_done():
            self.env.render()

            action = self.select_action(state)

            next_state, reward, done, _ = self.env.step(action)

            state = next_state

            if done[0]:
                print("score :", self.env.scores[0])

    def write_log(self, global_step: int, **kargs):
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
            elif isinstance(value, (float, np.float)):
                s += f"{name} : {value:.3f} | "
            else:
                s += name + " : " + str(value)
            self.writer.add_scalar(name, value, global_step)

        print(s)
    
        wandb.log(kargs)

