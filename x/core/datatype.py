from abc import ABC, abstractmethod


class AExecution(ABC):
    r"""
    Abstract class for execution
    (avoid circular dependencies for registr)
    """

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def eval(self, *args, **kwargs):
        pass

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def register_all(self, *args, **kwargs):
        pass


class ACheckpointManager(ABC):
    r"""
    Abstract class for checkpointing manager
    (avoid circular dependencies for registr)
    """

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_checkpoint(self, *args, **kwargs):
        pass

    @abstractmethod
    def model_state_dict(self, *args, **kwargs):
        pass


class ALRScheduler(ABC):
    r"""
    Abstract class for learning scheduler
    (avoid circular dependencies for registr)
    """

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abstractmethod
    def update_lr(self, *args, **kwargs):
        pass
