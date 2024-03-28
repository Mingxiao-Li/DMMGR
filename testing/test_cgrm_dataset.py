import os, json
import sys

pt_path = "/export/home2/NoCsBack/hci/mingxiao/vqa"
module_path = os.path.abspath(pt_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from x import *
from torch.utils.data import DataLoader
import pdb
import torch
from model_zoo.graph_models.graph_dataset import GraphDataset


if __name__ == "__main__":
    config_path = "experiment_configs/test_dataset_config.yaml"
    _C = XCfgs(config_path)
    _C.proc()
    config = _C.get_config()
    data_config = config.DATA

    dataset = GraphDataset(data_config)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )

    for i, item in enumerate(data_loader):
        question = item["question"]
        question_mask = item["question_mask"]
        answer = item["answer"]
        img_feature = item["img_nodes_feature"]
        img_loc = item["img_edges_feature"]
        img_node1_ids_list = item["img_node1_ids_list"]
        img_node2_ids_list = item["img_node2_ids_list"]
        kg_entity_tensor = item["kg_entity_tensor"]
        kg_edge_tensor = item["kg_edge_tensor"]
        kg_node1_ids_list = item["kg_node1_ids_list"]
        kg_node2_ids_list = item["kg_node2_ids_list"]
        # q_node1_ids_list = item["lang_node1_ids_list"]
        # q_node2_ids_list = item["lang_node2_ids_list"]
    print("DONE")