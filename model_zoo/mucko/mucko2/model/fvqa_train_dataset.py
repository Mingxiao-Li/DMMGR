from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import json
import numpy as np
import pickle
import torch
from util.vocabulary import Vocabulary
from torch.utils.data import DataLoader
import dgl
from math import sqrt, atan2
import yaml
import h5py


class FvqaTrainDataset(Dataset):
    def __init__(self, config, overfit=False, in_memory=True):
        super().__init__()

        self.config = config
        self.overfit = overfit
        self.qids = []
        self.questions = []
        self.question_lengths = []
        self.image_ids = []

        self.fact_num_nodes = []
        self.fact_e1ids_list = []
        self.fact_e2ids_list = []
        self.fact_answer_ids = []
        self.fact_answer_list = []

        self.semantic_num_nodes = []
        self.semantic_e1ids_list = []
        self.semantic_e2ids_list = []

        self.image_features = []
        self.image_relations = []
        self.image_files = []

       
        with open("fvqa_data/36bbox_res.pickle", 'rb') as f:
            self.bboxs = pickle.load(f)

        print('loading qa raw...')
        with open('fvqa_data/train/train_qa_raw.pickle', 'rb') as f:
            qa_raw = pickle.load(f)

        print('loading semantic graph feature...')
        sem_data = np.load('fvqa_data/train/semantic_graph_train_feature.npz')
        self.semantic_graph_node_features = sem_data['node_features']
        self.semantic_graph_edge_features = sem_data['edge_features']

        print('loading fact graph feature....')
        fact_data = np.load('fvqa_data/train/fact_graph_train_feature.npz')
        self.fact_graph_node_features = fact_data['node_features']
        self.fact_graph_edge_features = fact_data['edge_features']

        for qid, qa_item in qa_raw.items():
            image_file = qa_item['img_file']
            self.qids.append(qid)
            self.questions.append(qa_item['question'])
            self.question_lengths.append(qa_item['question_length'])
            self.image_files.append(image_file)

            self.fact_num_nodes.append(qa_item['fact_graph']['num_node'])
            self.fact_e1ids_list.append(qa_item['fact_graph']['e1ids'])
            self.fact_e2ids_list.append(qa_item['fact_graph']['e2ids'])
            self.fact_answer_ids.append(qa_item['fact_graph']['answer_id'])

            self.semantic_num_nodes.append(qa_item['semantic_graph']['num_node'])
            self.semantic_e1ids_list.append(qa_item['semantic_graph']['e1ids'])
            self.semantic_e2ids_list.append(qa_item['semantic_graph']['e2ids'])

        if overfit:
            self.qids = self.qids[:100]
            self.questions = self.questions[:100]
            self.question_lengths = self.question_lengths[:100]

            self.fact_num_nodes = self.fact_num_nodes[:100]
            self.fact_e1ids_list = self.fact_e1ids_list[:100]
            self.fact_e2ids_list = self.fact_e2ids_list[:100]
            self.fact_answer_ids = self.fact_answer_ids[:100]
            self.fact_graph_node_features = self.fact_graph_node_features[:100]

            self.semantic_num_nodes = self.semantic_num_nodes[:100]
            self.semantic_e1ids_list = self.semantic_e1ids_list[:100]
            self.semantic_e2ids_list = self.semantic_e2ids_list[:100]
            self.semantic_graph_node_features = self.semantic_graph_node_features[:100]
            self.semantic_graph_edge_features = self.semantic_graph_edge_features[:100]

            self.image_features = self.image_features[:100]
            self.image_relations = self.image_relations[:100]

    def __getitem__(self, index):

        item = {}
        item['id'] = self.qids[index]
        item['question'] = torch.Tensor(self.questions[index]).long()
        item['question_length'] = self.question_lengths[index]

        image_file = self.image_files[index]
        w = self.bboxs[image_file]['image_w']
        h = self.bboxs[image_file]['image_h']
        image_feature = torch.tensor(self.bboxs[image_file]['features'])
        # 图像归一化
        if self.config['dataset']["img_norm"]:
            image_feature = normalize(image_feature, dim=0, p=2)

        img_bboxes = self.bboxs[image_file]['boxes']
        img_rel = np.zeros((36, 36, 7))
        for i in range(36):
            for j in range(36):
                xi = img_bboxes[i][0]
                yi = img_bboxes[i][1]
                wi = img_bboxes[i][2]
                hi = img_bboxes[i][3]
                xj = img_bboxes[j][0]
                yj = img_bboxes[j][1]
                wj = img_bboxes[j][2]
                hj = img_bboxes[j][3]

                r1 = (xj - xi) / (wi * hi) ** 0.5
                r2 = (yj - yi) / (wi * hi) ** 0.5
                r3 = wj / wi
                r4 = hj / hi
                r5 = (wj * hj) / wi * hi
                r6 = sqrt((xj - xi) ** 2 + (yj - yi) ** 2) / sqrt(w ** 2 + h ** 2)
                r7 = atan2(yj - yi, xj - xi)

                rel = [r1, r2, r3, r4, r5, r6, r7]
                img_rel[i][j] = rel

        item['img_features'] = image_feature
        item['img_relations'] = torch.Tensor(img_rel)

        item['facts_num_nodes'] = self.fact_num_nodes[index]
        item['facts_node_features'] = torch.Tensor(self.fact_graph_node_features[index])
        item['facts_edge_features'] = torch.Tensor(self.fact_graph_edge_features[index])
        item['facts_e1ids'] = torch.Tensor(self.fact_e1ids_list[index]).long()
        item['facts_e2ids'] = torch.Tensor(self.fact_e2ids_list[index]).long()
        item['facts_answer_id'] = self.fact_answer_ids[index]
        # item['facts_answer'] = torch.Tensor(self.fact_answer_list[index])

        answer = np.zeros(self.fact_num_nodes[index])
        if self.fact_answer_ids[index] != -1:
            answer[self.fact_answer_ids[index]] = 1
        item['facts_answer'] = torch.Tensor(answer)

        item['semantic_num_nodes'] = self.semantic_num_nodes[index]
        item['semantic_node_features'] = torch.Tensor(self.semantic_graph_node_features[index])
        item['semantic_edge_features'] = torch.Tensor(self.semantic_graph_edge_features[index])
        item['semantic_e1ids'] = torch.Tensor(self.semantic_e1ids_list[index]).long()
        item['semantic_e2ids'] = torch.Tensor(self.semantic_e2ids_list[index]).long()

        return item

    def __len__(self):
        if (self.overfit):
            return 100
        else:
            return len(self.qids)
