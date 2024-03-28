from x.core.registry import registry
from x.core.datatype import AExecution
import torch
import os, logging


class XExecution(AExecution):
    """
    Base class for Execution class
    All Execution class should inherent from this base class
    """

    def __init__(self, _C):
        self._C = _C

        self.register_all()

        # get logger
        print("Getting log")
        _log_init = registry.get_logger("logger")
        self.logger = _log_init(
            name="logger", level=logging.INFO, format_str="%(asctime)s- %(message)s"
        )
        if _C.EXECUTION.log_path is not None:
            self.log_path = _C.EXECUTION.log_path.format(version=_C.EXECUTION.version)
            self.logger.add_filehandler(self.log_path)

        self.logger.info("Loading data ......")
        _dataset_init = registry.get_dataset(_C.DATA.name)
        self.dataset = _dataset_init(_C.DATA)
        self.logger.info(
            "Num of samples of {} dataset :{}".format(_C.DATA.split, len(self.dataset))
        )

        if _C.EXECUTION.is_valid:
            self.logger.info("Loading eval data ......")
            _C.defrost()
            _C.DATA.split = "val"
            _C.freeze()
            self.eval_dataset = _dataset_init(_C.DATA)
            self.logger.info(
                "Num of samples of eval dataset :{}".format(len(self.eval_dataset))
            )
        else:
            self.eval_dataset = None

        # set device
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if _C.OPTIM.lr_scheduler is not None:
            self._lr_scheduler = registry.get_lrscheduler("LRScheduler")

        # set checkpoint
        self._checkpointing = registry.get_checkpointing("CheckpointManager")

        self.logger.info(self._C)

    def register_all(self):
        """
        Register all necessary components (model,dataset,...) to run the experiments.
        If registry decorater is used in model/dataset class, then only import the class here.
        Example
        --------
        try:
             from model_dir import Model
             from data_dir import Dataset
        except:
            raise ImportError
        --------
        """
        raise NotImplementedError

    def train(self, dataset, eval_dataset=None):
        raise NotImplementedError

    def eval(self, dataset, state_dict_path=None, valid=None):
        raise NotImplementedError

    def run(self, run_mode):
        if run_mode == "train":
            if self._C.EXECUTION.is_valid:
                self.train(self.dataset, self.eval_dataset)
            else:
                self.train(self.dataset, None)

        elif run_mode == "val":
            self.eval(self.dataset, valid=True)

        elif run_mode == "test":
            assert (
                self._C.DATA.split == "test"
            ), "Data split is {}. Split has to be 'test' during testing !!".format(
                self._C.DATA.split
            )
            self.eval(self.dataset)
        
