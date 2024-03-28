from typing import Optional, Union, List
from types import MethodType
import yacs.config
import os, torch, random
import numpy as np


class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config
CONFIG_FILE_SEPARATOR = ","

_C = CN()

# ----------
# Env
# ----------
_C.ENV = CN()
_C.ENV.gpus = "1,2"
_C.ENV.seed = random.randint(0, 99999999)

# ----------
# EXECUTION
# ----------
_C.EXECUTION = CN()
_C.EXECUTION.is_valid = True
_C.EXECUTION.run_mode = "train"
_C.EXECUTION.n_gpu = 1
_C.EXECUTION.resume = False
_C.EXECUTION.load_checkpoint_path = None
_C.EXECUTION.version = "exp"
_C.EXECUTION.log_path = "train_info_{version}"
# ----------
# Data
# ----------
_C.DATA = CN()
_C.DATA.parent_path = "/cw/liir/NoCsBack/testliir/datasets/"
_C.DATA.batch_size = 32
_C.DATA.split = "train"
_C.DATA.num_workers = 4
_C.DATA.name = "dataset"
_C.DATA.max_seq_len = 128
_C.DATA.num_train_sample = 0
_C.DATA.num_val_sample = 0
_C.DATA.load_test_set = False
_C.DATA.shuffle = True

# ----------
# MODEL
# ----------
_C.MODEL = CN()

# ----------
# OPTIMIZER
# ----------
_C.OPTIM = CN()
_C.OPTIM.lr_base = 1e-3
_C.OPTIM.lr_scheduler = "warmup"
_C.OPTIM.max_epoch = 30


class XCfgs:
    def __init__(
        self,
        config_paths: str = None,
    ):
        r"""Create a unified config with default values overwritten by values from
        :p:`config_paths` and overwritten by options from :p:`opts`.
        :param config_paths: List of config paths or string that contains comma
            separated list of config paths.
        """
        self.config = _C.clone()
        if config_paths:
            if isinstance(config_paths, str):
                if CONFIG_FILE_SEPARATOR in config_paths:
                    config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
                else:
                    config_paths = [config_paths]

                for config_path in config_paths:
                    self.config.merge_from_file(config_path)
            self.config.freeze()

    def proc(self):
        # set up environment

        # ---------- Devices setup ---------
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.ENV.gpus
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        num_gpu = len(self.config.ENV.gpus.split(","))
        self.config.defrost()
        self.config.EXECUTION.n_gpu = num_gpu
        self.config.freeze()
        # ---------- Seed setup ----------
        # fix pytorch seed
        torch.manual_seed(self.config.ENV.seed)
        if num_gpu < 2:
            torch.cuda.manual_seed(self.config.ENV.seed)
        else:
            torch.cuda.manual_seed_all(self.config.ENV.seed)
        torch.backends.cudnn.deterministic = True

        # fix numpy seed
        np.random.seed(self.config.ENV.seed)

        # fix random seed
        random.seed(self.config.ENV.seed)

    def add_from_args(self, args: dict) -> None:
        """
        :param args_dict:  dictionary contain parameters needed to be updata in config
        :return:
        """
        args_dict = self.parse_to_dict(args)
        self.config.defrost()
        args_list = []
        for arg in args_dict:
            args_list.extend([arg, args_dict[arg]])
        self.config.merge_from_list(args_list)
        self.config.freeze()

    def parse_to_dict(self, args) -> dict:
        """
        :param args: NameSpace from command line
        :return:
        """
        args_dict = {}
        for arg in dir(args):
            if not arg.startswith("_") and not isinstance(
                getattr(args, arg), MethodType
            ):
                if getattr(args, arg) is not None:
                    v = getattr(args, arg)
                    if type(v) is str:
                        # if the value is str, it has to be this format "''"
                        # as config.merge_from_lst will convert str("...") to literal level
                        args_dict[arg] = '"' + v + '"'
                    else:
                        args_dict[arg] = v
        return args_dict

    def get_config(self) -> Config:
        return self.config
