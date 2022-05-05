from abc import ABC, abstractmethod
class AudioModel(ABC):
    @abstractmethod
    def __call__(self, inputs):
        raise NotImplementedError()
    @abstractmethod
    def zero_grad(self):
        raise NotImplementedError()