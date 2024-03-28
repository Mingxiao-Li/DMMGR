import collections, logging
from typing import Any, Callable, DefaultDict, Optional, Type, Dict
from torch.utils.data import Dataset
from x.core.datatype import AExecution, ACheckpointManager, ALRScheduler
import torch.nn as nn


class Singleton(type):
    _instances: Dict["Singleton", "Singleton"] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class XRegistry(metaclass=Singleton):
    # A global registry
    mapping: DefaultDict[str, Any] = collections.defaultdict(dict)

    @classmethod
    def _register_impl(
        cls,
        _type: str,
        to_register: Optional[Any],
        name: Optional[str],
        assert_type: Optional[Type] = None,
    ) -> Callable:
        def wrap(to_register):
            if assert_type is not None:
                assert issubclass(
                    to_register, assert_type
                ), "{} must be a subclass of {}".format(to_register, assert_type)
            register_name = to_register.__name__ if name is None else name
            cls.mapping[_type][register_name] = to_register
            return to_register

        if to_register is None:
            return wrap
        else:
            return wrap(to_register)

    @classmethod
    def register_dataset(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a dataset to registry with key:p:'name'
        :param to_register:
        :param name: Key with which the dataset will be registered.
                If: 'None' will use the name of the class
        Example
        --------
        >>>  @register.register_dataset(name="MyDatasetName")
        >>>  class MyDatast(Dataset):
        >>>      pass
        """
        return cls._register_impl("dataset", to_register, name, assert_type=Dataset)

    @classmethod
    def register_model(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a model to registry with key :p:'name'
        :param name: Key with which the task will be registered.
            If :py:`None` will use the name of the class:
        """
        return cls._register_impl("model", to_register, name, assert_type=nn.Module)

    @classmethod
    def register_execution(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register a Execution to registry with key :p:'name'
        :param name: Key with which the task will be registered.
            If :py:`None` will use the name of the class:
        """
        return cls._register_impl(
            "execution", to_register, name, assert_type=AExecution
        )

    @classmethod
    def register_checkpointing(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register the checkpointing to registry with key :p:'name'
        :param name: Key with which the task will be registered.
            If :py:`None` will use the name of the class:
        """
        return cls._register_impl(
            "checkpointing", to_register, name, assert_type=ACheckpointManager
        )

    @classmethod
    def register_lrscheduler(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register the learning rate scheduler to registry with key :p:'name'
        :param name: Key with which the task will be registered.
            If :py:`None` will use the name of the class:
        """
        return cls._register_impl(
            "lrscheduler", to_register, name, assert_type=ALRScheduler
        )

    @classmethod
    def register_logger(cls, to_register=None, *, name: Optional[str] = None):
        r"""Register the logger to registry with key :p:'name'
        :param name: Key with which the task will be registered.
                If :py:`None` will use the name of the class:
        """
        return cls._register_impl(
            "logger", to_register, name, assert_type=logging.Logger
        )

    @classmethod
    def register(cls, type: str, to_register=None, *, name: Optional[str] = None):
        r"""Register new type that is not defined in this class
        :param name: Key with which the task will be registered.
        If :py:`None` will use the name of the class:
        """
        return cls._register_impl(type, to_register, name, assert_type=None)

    @classmethod
    def _get_impl(cls, _type: str, name: str) -> Type:
        return cls.mapping[_type].get(name, None)

    @classmethod
    def get_dataset(cls, name: str) -> [Dataset]:
        return cls._get_impl("dataset", name)

    @classmethod
    def get_model(cls, name: str) -> [nn.Module]:
        return cls._get_impl("model", name)

    @classmethod
    def get_execution(cls, name: str) -> [AExecution]:
        return cls._get_impl("execution", name)

    @classmethod
    def get_lrscheduler(cls, name: str) -> [ALRScheduler]:
        return cls._get_impl("lrscheduler", name)

    @classmethod
    def get_checkpointing(cls, name: str) -> [ACheckpointManager]:
        return cls._get_impl("checkpointing", name)

    @classmethod
    def get_logger(cls, name: str) -> [logging.Logger]:
        return cls._get_impl("logger", name)

    @classmethod
    def get(cls, type: str, name: str) -> Any:
        return cls._get_impl(type, name)


registry = XRegistry()
