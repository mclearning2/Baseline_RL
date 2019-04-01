from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def train_model(self, *args):
        pass

    @abstractmethod
    def write_log(self, *args):
        pass

    @abstractmethod
    def train(self, config):
        pass

    @abstractmethod
    def test(self, config):
        pass
