from x.core.registry import registry
from x.core.dataset import XDataset
from x.common.vocabulary import Vocabulary
from x.common.image_feature_reader import ImageFeaturesH5Reader
from model_zoo.reason_flow_net.utils import get_reason_graph,get_knowledge_graph
from model_zoo.reason_flow_net.utils import get_img_connection
import numpy as np
import torch
import os
import json


@registry.register_dataset(name="RFDataset")
class RFDataset(XDataset):

    def __init__(self, _C):
        super(RFDataset,self).__init__(_C)
        self.config = _C
        self.max_entity_len = _C.max_entity_len
        self.max_num_kg = _C.max_num_kg

        candidate_answer_path = os.path.join(_C.parent_path, _C.candidate_answer_path)
        with open(candidate_answer_path,"r") as f:
            self.candidate_answers = json.load(f)["answers"]

        with open(os.path.join(_C.parent_path, _C.word_count_path), "r") as f:
            word_count =json.load(f)

        kg_to_lang_path = os.path.join(_C.parent_path, _C.kg_to_lang_path)
        with open(kg_to_lang_path, "r") as f:
            self.kg_to_lang = json.load(f)

        self.vocabulary = Vocabulary(word_count, min_count=0)
        self.image_feature_reader = ImageFeaturesH5Reader(_C.image_feature_path)
        self.k_nearest_node = _C.k_nearest_node # set the number of nodes, one image node connect to
        assert self.k_nearest_node <= 36, "The maxmium k is 36!!"

        self._data = self.filter_dataset()

    def encode_kg(self, entity, e1_ids_list, e2_ids_list, edges):
        entity_tensor = torch.ones(self.max_num_kg*2, self.max_entity_len)
        edges_tensor = torch.ones(self.max_num_kg, self.max_entity_len)
        e1_ids_list_tensor = torch.ones(self.max_num_kg*2) * -1
        e2_ids_list_tensor = torch.ones(self.max_num_kg*2) * -1
        e1_ids_list_tensor[:len(e1_ids_list)] = torch.tensor(e1_ids_list)
        e2_ids_list_tensor[:len(e2_ids_list)] = torch.tensor(e2_ids_list)
        for j,e in enumerate(entity):
            e_ids = self.vocabulary.to_indices(e[0].split())
            entity_tensor[j,:len(e_ids)] = torch.tensor(e_ids)
        for i,e in enumerate(edges):
            e_ids = self.vocabulary.to_indices(e[0].split())
            edges_tensor[i,:len(e_ids)] = torch.tensor(e_ids)
        return entity_tensor, e1_ids_list_tensor, e2_ids_list_tensor, edges_tensor

    def encode_reason_flow(self, total_nodes, total_connect, note_type):
        total_nodes_tensor = torch.ones(5, self.max_entity_len)
        total_connect_tensor = torch.ones(5)
        note_type_tensor = torch.zeros(5)
        for i,node in enumerate(total_nodes):
            node = self.vocabulary.to_indices(node[0].split())
            total_nodes_tensor[i,:len(node)] = torch.tensor(node)
        total_connect_tensor[:len(total_nodes)] = torch.tensor(total_connect)
        note_type_tensor[:len(note_type)] = torch.tensor(note_type)
        return total_nodes_tensor, total_connect_tensor, note_type_tensor

    def encode_images(self,features, num_boxes, boxes, node1_id_list, node2_id_list,max_region=37):
        num_boxes = min(int(num_boxes),max_region)
        mix_boxes_pad = np.zeros((max_region, boxes.shape[-1]))
        mix_feature_pad = np.zeros((max_region, features.shape[-1]))

        mix_boxes_pad[:num_boxes] = boxes[:num_boxes]
        mix_feature_pad[:num_boxes] = features[:num_boxes]

        features = torch.tensor(mix_feature_pad, dtype=torch.float32)
        img_loc = torch.tensor(mix_boxes_pad, dtype=torch.float32)
        node1_id_list = torch.tensor(node1_id_list, dtype=torch.float32)
        node2_id_list = torch.tensor(node2_id_list, dtype=torch.float32)

        return features, img_loc, node1_id_list, node2_id_list

    def __getitem__(self, index):
        item = {}
        current_data = self._data[index]
        img_id = current_data["image_id"]

        total_nodes, total_connect, node_type = get_reason_graph(self.config.RF_SETUP,current_data,self.kg_to_lang)
        total_node_tensor,total_connect_tensor,type_tensor = self.encode_reason_flow(total_nodes, total_connect,
                                                                      node_type)

        # get kg info
        entity, e1_ids_list, e2_ids_list, edges = get_knowledge_graph(self.config.KG_SETUP, current_data, self.kg_to_lang)
        entity_tensor, e1_ids_list_tensor, e2_ids_list_tensor, edges_tensor = self.encode_kg(entity, e1_ids_list, e2_ids_list, edges)

        #get image info
        feature, num_boxes, boxes, _, _   = self.image_feature_reader[img_id]
        # boxes: normalized box position [whole picture, x1, y1,x2,y2,ratio] 36*6
        node1_id_list, node2_id_list = get_img_connection(boxes, num_boxes, self.k_nearest_node)
        img_feature, img_loc, img_node1_ids_list, img_node2_ids_list = self.encode_images(feature, num_boxes, boxes,
                                                                                          node1_id_list,node2_id_list)
        # shape 36 * k_nearest

        item["kg_entity"] = entity_tensor     # batch, max_num_kg*2, max_entity_len
        item["kg_e1_ids_list"] = e1_ids_list_tensor   # batch, max_num_kg*2
        item["kg_e2_ids_list"] = e2_ids_list_tensor    # batch, max_num_kg*2
        item["kg_edge"] = edges_tensor
        item["r_nodes"] = total_node_tensor  # batch, 5, max_entity_len
        item["r_connects"] = total_connect_tensor # batch, 5, max_entity_len
        item["r_type"] = type_tensor  # batch, 5
        item["img_feat"] = img_feature   # batch, img_size
        item["img_node1_ids_list"] = img_node1_ids_list #batch, num_box * k_nearest_node
        item["img_node2_ids_list"] = img_node2_ids_list # batch, num_box * k_nearest_node
        item["img_loc"] = img_loc  # batch,6
        return item


    def filter_dataset(self):
        print("Filtering ......")
        _data = []
        if self._config.only_kb_related:
            for data in self._data:
                if data["KB"] == 1:
                    _data.append(data)

        if self._config.only_kb_not_related:
            for data in self._data:
                if data["KB"] == 0:
                    _data.append(data)

        if self._config.only_q_type is not None:
            for data in self._data:
                if data["qtype"] not in self._config.only_q_type:
                    _data.append(data)
        del self._data
        return _data
