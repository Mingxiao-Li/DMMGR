from x import *
from torch.utils.data import DataLoader
import os, json
import pdb
import torch
from model_zoo.graph_models.graph_dataset import GraphDataset


if __name__ == "__main__":
    config_path = "experiment_configs/test_dataset_configs.yaml"
    _C = XCfgs(config_path)
    _C.proc()
    config = _C.get_config()
    data_config = config.DATA

    data_path = os.path.join(
        data_config.parent_path, data_config.path.format(split="val")
    )

    dataset = GraphDataset(data_config)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )

    for i, item in enumerate(data_loader):
        question = item["question"]
        batch_size, _ = question.shape
        question_mask = item["question_mask"]
        answer = item["answer"]
        img_nodes = item["img_nodes_feature"]
        img_loc = item["img_edges_feature"]
        img_node1_ids_list = item["img_node1_ids_list"]
        img_node2_ids_list = item["img_node2_ids_list"]
        kg_entity_tensor = item["kg_entity_tensor"]
        kg_edge_tensor = item["kg_edge_tensor"]
        kg_node1_ids_list = item["kg_node1_ids_list"]
        kg_node2_ids_list = item["kg_node2_ids_list"]
        q_node1_ids_list = item["lang_node1_ids_list"]
        q_node2_ids_list = item["lang_node2_ids_list"]

        num_img_nodes = torch.sum(torch.sum(img_nodes, dim=2) != 10, dim=1)
        for b in range(batch_size):
            if num_img_nodes[b] == 0:
                print(item["img_id"][b])
