from x import *
from model_zoo.mcan.exec import MCANExecution
import argparse


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--BS", dest="DATA.batch_size", help="batch size during training,", type=int
    )

    parser.add_argument(
        "--NW",
        dest="DATA.num_workers",
        default=4,
        help="multithreaded loading",
        type=int,
    )

    parser.add_argument(
        "--SPLIT",
        dest="DATA.split",
        default="train",
        choices=["train", "valid", "test"],
        help="set dataset split,",
        type=str,
    )

    parser.add_argument(
        "--NUM_TRAIN",
        dest="DATA.num_train_sample",
        default=0,
        help="how many training samples are used. (for testing or overfit)"
        "Note: 0 means all.",
        type=int,
    )

    parser.add_argument(
        "--NUM_VAL",
        dest="DATA.num_val_sample",
        default=0,
        help="how many validation samples are used. (for testing or overfit)"
        "Note: 0 means all",
        type=int,
    )

    parser.add_argument(
        "--GPU", dest="ENV.gpus", default="3", help="gpu select, eg.'0,1,2'", type=str
    )

    parser.add_argument(
        "--RUN_MODE",
        dest="EXECUTION.run_mode",
        default="train",
        choices=["train", "val", "test"],
        help="set runing model (train,val,test)",
        type=str,
    )

    parser.add_argument(
        "--IS_VALID",
        dest="EXECUTION.is_valid",
        default=True,
        help="set if validate after each epoch",
        type=bool,
    )

    parser.add_argument(
        "--MAX_EPOCH",
        dest="OPTIM.max_epoch",
        default=40,
        help="max training epoch",
        type=int,
    )

    parser.add_argument(
        "--CP", dest="_config_path", help="path to config file", type=str
    )

    _args = parser.parse_args()
    return _args


if __name__ == "__main__":
    args = parse_args()
    args._config_path = "experiment_configs/mcan_small.yaml"
    _C = XCfgs(args._config_path)
    _C.add_from_args(args)
    _C.proc()

    config = _C.get_config()
    execution = registry.get_execution(config.EXECUTION.name)
    execution = execution(config)
    execution.run(config.EXECUTION.run_mode)
