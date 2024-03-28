from x import *
from model_zoo.graph_models.graph_excute import CGRMExecution
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
        default=0,
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
        choices=["train", "val", "test", "analysis"],
        help="set runing model (train,val,test)",
        type=str,
    )

    parser.add_argument(
        "--IS_VALID",
        dest="EXECUTION.is_valid",
        help="set if validate after each epoch",
        type=bool,
    )

    parser.add_argument(
        "--SF",
        dest="DATA.shuffle",
        help="set if shuffle dataloader",
        type=bool,
    )

    parser.add_argument(
        "--MAX_EPOCH",
        dest="OPTIM.max_epoch",
        default=60,
        help="max training epoch",
        type=int,
    )

    parser.add_argument(
        "--CP", dest="_config_path", help="path to config file", type=str
    )

    parser.add_argument(
        "--VALID_BS",
        dest="DATA.valid_batch_size",
        help="batch size of validation",
        type=int,
    )

    parser.add_argument(
        "--VERSION",
        dest="EXECUTION.version",
        help="experiment info",
        type=str,
    )

    parser.add_argument(
        "--MODEL_NAME",
        dest="MODEL.name",
        help="the name of the model",
        type=str,
    )

    parser.add_argument(
        "--MODEL_LT",
        dest="MODEL.graph_layer_type",
        help="specific the layer typr of gat",
        type=str,
    )

    parser.add_argument(
        "--A_DATA",
        dest="ANALYSIS.dataset",
        type=str,
    )

    parser.add_argument(
        "--A_OUT", dest="ANALYSIS.output_path", help="path to save results", type=str
    )

    parser.add_argument(
        "--A_checkpoint",
        dest="ANALYSIS.state_dict_path",
        help="checkpoint need to be analyzed",
        type=str,
    )

    _args = parser.parse_args()

    return _args


if __name__ == "__main__":
    args = parse_args()
    # args._config_path = "experiment_configs/graph_model.yaml"
    _C = XCfgs(args._config_path)
    _C.add_from_args(args)
    _C.proc()

    config = _C.get_config()

    execution = registry.get_execution(config.EXECUTION.name)
    execution = execution(config)
    execution.run(config.EXECUTION.run_mode)