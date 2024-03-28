import torch
import os
import json
import numpy as np
from x.core.registry import registry
from x.core.dataset import XDataset
from x.common.util import get_word_count
from x.common.vocabulary import Vocabulary
from x.common.image_feature_reader import ImageFeaturesH5Reader
from model_zoo.graph_models.graph_generator import (
    ImgSceneGraphGetter,
    KGGraphGetter,
    ImgGraphGetter,
    LanguageGraphGetter,
    put_answer_in_graph,
)
import h5py
import pdb
import re


EMPTY_SG_IMG_ID = [
    51,
    61,
    157,
    242,
    285,
    300,
    318,
    372,
    404,
    513,
    521,
    648,
    678,
    746,
    1103,
    1117,
    1137,
    1213,
    1240,
    1361,
    1666,
    1889,
    1912,
    2526,
    2579,
    2614,
    2655,
    2775,
    3443,
    3674,
    3727,
    3755,
    3900,
    3946,
    4052,
    4088,
    4216,
    4410,
    4415,
    4565,
    4567,
    4755,
    4780,
    4793,
    61553,
    107937,
    107963,
    150288,
    150523,
    285628,
    285793,
    285800,
    498344,
    712976,
    713106,
    713274,
    713545,
    713671,
    1159691,
    1159966,
    1160243,
    1591927,
    1592136,
    1592564,
    1592880,
    2414824,
    2414195,
    2414028,
    2404402,
    2404292,
    2403678,
    2394423,
    2390392,
    2390119,
    2384670,
    2384264,
    2384018,
    2383960,
    2381096,
    2379875,
    2372747,
    2366399,
    2344520,
    2336999,
    2414055,
    2416187,
    2320768,
]


@registry.register_dataset(name="GraphDataset")
class GraphDataset(XDataset):
    def __init__(self, config):
        super(GraphDataset, self).__init__(config)
        self.config = config
        self.k_nearest = config.k_nearest
        self.top_k_kg = config.top_k_kg
        self.ori_region_loc = config.ori_region_loc
        self.load_retrieval_kg = config.load_retrieval_kg
        self.use_lang_graph = config.use_lang_graph
        self.use_scene_graph = config.use_scene_graph
        self.put_answer_in_graph = config.put_answer_in_graph
        self.use_split_answer_sets = config.use_split_answer_sets

        if not self.use_split_answer_sets:
            candidated_answer_path = os.path.join(
                config.parent_path, config.candidate_answer_path
            )
            with open(candidated_answer_path, "r") as f:
                self.candidate_answers = json.load(f)["answers"]
        else:
            e_candidate_answer_path = os.path.join(
                config.parent_path, config.candidate_answer_e_path
            )
            r_candidate_answer_path = os.path.join(
                config.parent_path, config.candidate_answer_r_path
            )

            with open(e_candidate_answer_path, "r") as f:
                self.candidate_answers_e = list(json.load(f).keys())
                assert len(self.candidate_answers_e) == 5390

            with open(r_candidate_answer_path, "r") as f:
                self.candidate_answers_r = list(json.load(f).keys())
                assert len(self.candidate_answers_r) == 1709

        # build vocabulary
        if config.word_count_path is None:
            word_count = get_word_count(self._data)
        else:
            word_count = get_word_count(
                os.path.join(config.parent_path, config.word_count_path)
            )

        self._vocabulary = Vocabulary(word_count, min_count=1)

        # write word2index to file
        word2index_path = os.path.join(
            self.config.parent_path, self.config.word2index_path
        )
        if not os.path.exists(word2index_path):
            with open(word2index_path, "w") as f:
                json.dump(self._vocabulary.word2index, f)

        if not config.use_scene_graph:
            # set image_feature reader
            if config.use_symbolic:
                object_path = os.path.join(config.parent_path, config.object_path)
                self.image_feature_reader = ImageFeaturesH5Reader(
                    config.img_feature_path, cls_prediction=True
                )
            else:
                self.image_feature_reader = ImageFeaturesH5Reader(
                    config.img_feature_path
                )
            self.img_graph_getter = ImgGraphGetter(
                self.image_feature_reader,
                config.max_region,
                config.k_nearest,
                config.use_symbolic,
                object_path,
                self._vocabulary,
            )
        else:
            if config.use_gt_sg:
                scene_graph_path = os.path.join(
                    config.parent_path, config.scene_graph_path
                )
                with open(scene_graph_path, "r") as f:
                    self.scene_graphs = json.load(f)

                self.img_graph_getter = ImgSceneGraphGetter(
                    self.scene_graphs,
                    self._vocabulary,
                    config.max_num_img_node,
                    config.max_img_node_len,
                    config.max_num_img_edge,
                    config.max_img_edge_len,
                )
            else:
                ge_sg_info_path = os.path.join(
                    config.parent_path, config.ge_scene_graph_info
                )
                self.ge_sg_path = os.path.join(
                    config.parent_path, config.ge_scene_graph
                )
                with open(ge_sg_info_path, "r") as f:
                    self.ge_scene_graph_info = json.load(f)
                self.scene_graphs = h5py.File(self.ge_sg_path, "r")
                self.img_graph_getter = ImgSceneGraphGetter(
                    self.scene_graphs,
                    self._vocabulary,
                    config.max_num_img_node,
                    config.max_img_node_len,
                    config.max_num_img_edge,
                    config.max_img_edge_len,
                    self.ge_scene_graph_info,
                )

            if self.put_answer_in_graph:
                synset_path = os.path.join(config.parent_path, config.synset_path)
                with open(synset_path, "r") as f:
                    self.ans_synsets = json.load(f)

            ### filter
            new_data = []
            for d in self._data:
                if d["image_id"] not in EMPTY_SG_IMG_ID:
                    new_data.append(d)
                    self._data = new_data

        # load kg
        all_kg_path = os.path.join(config.parent_path, config.kg_path)
        with open(all_kg_path, "r") as f:
            self.all_kg = json.load(f)["facts"]

        # if laod retrieval kg from file
        if self.load_retrieval_kg:
            retrieval_kg_path = os.path.join(
                config.parent_path, config.retrieval_kg_path
            )
            with open(retrieval_kg_path, "r") as f:
                self.retrieval_kg = json.load(f)

        # get kg_to_lange
        kg_to_lange_path = os.path.join(config.parent_path, config.kg_to_lang_path)
        with open(kg_to_lange_path, "r") as f:
            self.kg_to_lang = json.load(f)

        self.kg_graph_getter = KGGraphGetter(
            self.all_kg,
            self.retrieval_kg,
            self._vocabulary,
            self.kg_to_lang,
            config.top_k_kg,
            config.max_kg_ele_len,
            config.max_kg_ele_len,
        )

        if self.use_lang_graph:
            self.connections = None
            if config.load_connection_from_file:
                connection_path = os.path.join(
                    config.parent_path,
                    config.connection_path.format(split=config.split),
                )
                with open(connection_path, "r") as f:
                    self.connections = json.load(f)
            self.lang_graph_getter = LanguageGraphGetter(
                max_num_edge=config.max_num_lang_graph_edge,
                connections=self.connections,
            )

    def __getitem__(self, index):
        current_data = self._data[index]
        item = {}
        img_id = current_data["image_id"]
        question_id = current_data["question_id"]
        question = current_data["question"]
        answer = current_data["answer"]
        question_list = self._vocabulary.to_indices(question.split())
        question_tensor, question_mask = self.pad_sequence(
            question_list, max_seq_len=self.config.max_seq_len, return_mask=True
        )

        if not self.use_split_answer_sets:
            answer_tensor = torch.tensor(self.candidate_answers.index(answer))
        else:
            q_type = current_data["qtype"]
            if q_type in [0, 3, 4]:  # relation type
                q_type_tensor = torch.tensor(0)
                answer_index = self.candidate_answers_r.index(answer)
            else:
                q_type_tensor = torch.tensor(1)
                answer_index = len(
                    self.candidate_answers_r
                ) + self.candidate_answers_e.index(answer)
            answer_tensor = torch.tensor(answer_index)
            item["q_type"] = q_type_tensor

        # get lange connection
        if self.use_lang_graph:
            (
                lang_node1_list_tensor,
                lang_node2_list_tensor,
            ) = self.lang_graph_getter.get_lang_graph(question, question_id)
            item["lang_node1_ids_list"] = lang_node1_list_tensor
            item["lang_node2_ids_list"] = lang_node2_list_tensor

        # get kg
        (
            kg_entity_tensor,
            kg_edge_tensor,
            kg_node1_list_tensor,
            kg_node2_list_tensor,
        ) = self.kg_graph_getter.get_kg_graph(question_id)

        # boxes region location after normalize/ boxes_ori original region location
        if self.use_scene_graph:
            if self.config.use_gt_sg:
                (
                    img_node_features,
                    img_edge_features,
                    img_node1_ids_list,
                    img_node2_ids_list,
                ) = self.img_graph_getter.get_img_graph(img_id)
            else:
                (
                    img_node_features,
                    img_edge_features,
                    img_node1_ids_list,
                    img_node2_ids_list,
                ) = self.img_graph_getter.get_img_graph(img_id)

            if self.put_answer_in_graph:
                (
                    img_graph_node_list,
                    img_graph_edge_list,
                ) = self.img_graph_getter.get_entity_and_edge_list(img_id)

                (
                    kg_graph_node_list,
                    kg_graph_edge_list,
                ) = self.kg_graph_getter.get_entity_and_edge_list(question_id)

                answer_tensor = put_answer_in_graph(
                    self.ans_synsets,
                    answer,
                    img_graph_node_list,
                    img_graph_edge_list,
                    kg_graph_node_list,
                    kg_graph_edge_list,
                )

        else:
            (
                img_node_features,
                img_edge_features,
                img_node1_ids_list,
                img_node2_ids_list,
            ) = self.img_graph_getter.get_img_graph(img_id)

        item["img_id"] = torch.tensor(img_id)
        item["question"] = question_tensor.squeeze()
        item["question_mask"] = question_mask.squeeze()
        item["answer"] = answer_tensor
        item["img_nodes_feature"] = img_node_features
        item["img_edges_feature"] = img_edge_features
        item["img_node1_ids_list"] = img_node1_ids_list
        item["img_node2_ids_list"] = img_node2_ids_list
        item["kg_entity_tensor"] = kg_entity_tensor
        item["kg_edge_tensor"] = kg_edge_tensor
        item["kg_node1_ids_list"] = kg_node1_list_tensor
        item["kg_node2_ids_list"] = kg_node2_list_tensor
        item["question_id"] = torch.tensor(current_data["question_id"])

        return item
