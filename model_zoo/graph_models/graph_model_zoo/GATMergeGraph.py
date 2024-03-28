import torch
import os
import json
import torch.nn as nn
import dgl
import pdb
import torch.nn.functional as F
from model_zoo.graph_models.graph_model_zoo.graph_units import GAT
from model_zoo.graph_models.graph_model_zoo.graph_modules import LanguageEncoder
from x.core.registry import registry
from x.common.util import get_numpy_word_embed
from model_zoo.graph_models.graph_model_zoo.net_utils import *
from torch.nn.functional import normalize

@registry.register_model(name="MergeGAT")
class GATMergeGraph(nn.Module):
    def __init__(self, config):
        super(GATMergeGraph, self).__init__()
        self.config = config
        self.max_img_node_len = config.max_img_node_len
        self.max_img_edge_len = config.max_img_edge_len
        self.max_kg_len = config.max_kg_len

        if not config.use_glove_emb:
            self.word_embedding = nn.Embedding(config.vocab_size, config.embeded_size)
        elif config.use_glove_emb:
            word2index_path = os.path.join(config.parent_path, config.word2index_path)
            assert os.path.exists(word2index_path)

            with open(word2index_path, "r") as f:
                word2index = json.load(f)

            pretrained_word_emb = get_numpy_word_embed(
                word2index,
                os.path.join(config.parent_path, config.pretrained_word_path),
            )

            num_words, word_dim = pretrained_word_emb.shape

            self.word_embedding = nn.Embedding(num_words, word_dim)
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_word_emb))
            self.word_embedding.weight.requires_grad = config.fine_tune

        self.language_encoder = LanguageEncoder(config.LanguageEncoder)
        if not self.config.use_scene_graph:
            self.edge_linear = nn.Linear(config.img_edge_dim, config.embeded_size)

        self.gat = GAT(
            in_dim=config.in_dim,
            rel_dim=config.in_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.hidden_dim,
            num_heads=config.num_heads,
            layer_type=config.graph_layer_type,
        )

        self.question_node_proj = nn.Linear(
            config.LanguageEncoder.hidden_size, config.out_dim
        )
        self.question_edge_proj = nn.Linear(
            config.LanguageEncoder.hidden_size, config.embeded_size
        )

        self.edge_prj = nn.Linear(config.embeded_size, config.out_dim)

        self.mid_prj = nn.Linear(2 * config.out_dim, 3 * config.out_dim)
        self.pre_prj = nn.Linear(3 * config.out_dim, config.num_answer)

    def forward(self, item):

        question = item["question"]
        question_mask = item["question_mask"]
        img_nodes = item["img_nodes_feature"]
        img_edges = item["img_edges_feature"]
        img_node1_ids_list = item["img_node1_ids_list"]
        img_node2_ids_list = item["img_node2_ids_list"]
        kg_nodes = item["kg_entity_tensor"]
        kg_edges = item["kg_edge_tensor"]
        kg_node1_ids_list = item["kg_node1_ids_list"]
        kg_node2_ids_list = item["kg_node2_ids_list"]

        device = question.device
        seq_word_emb = self.word_embedding(question)
        seq_len = torch.sum(question_mask == 1, dim=1)
        all_out, lens, (hidden_out, cell_out) = self.language_encoder(
            seq_word_emb, seq_len
        )
        question_embed = hidden_out

        batch_size, _ = question.shape

        kg_entity_feat, kg_edge_feat,num_entities, num_edges = get_graph_feat(
            kg_nodes,
            kg_edges,
            self.word_embedding,
            self.max_kg_len,
            self.max_kg_len,
            True
        )
        node_embedding, edge_embedding, num_nodes_img, num_edges_img = get_graph_feat(
            img_nodes,
            img_edges,
            self.word_embedding,
            self.max_img_node_len,
            self.max_img_edge_len,
            self.config.use_scene_graph,
        )
        if not self.config.use_scene_graph:
            edge_embedding = self.edge_linear(edge_embedding)
            edge_embedding = normalize(edge_embedding)

        graph_batch, _ = build_and_merge_graph(
            batch_size = batch_size,
            num_img_nodes = num_nodes_img,
            num_img_edges = num_edges_img,
            img_node_feat = node_embedding,
            img_edge_feat = edge_embedding,
            img_node1_ids_list = img_node1_ids_list,
            img_node2_ids_list = img_node2_ids_list,
            num_kg_nodes = num_entities,
            num_kg_edges = num_edges,
            kg_node_feat = kg_entity_feat,
            kg_edge_feat = kg_edge_feat,
            kg_node1_ids_list = kg_node1_ids_list,
            kg_node2_ids_list = kg_node2_ids_list,
        )
        
        graph_batch = self.gat(graph_batch)
        graphs = dgl.unbatch(graph_batch)

        edges = []
        nodes = []
        for i in range(batch_size):
            nodes.append(torch.sum(graphs[i].ndata["feat"], dim=0).unsqueeze(0))
            edges.append(torch.sum(graphs[i].edata["feat"], dim=0).unsqueeze(0))
        nodes = torch.cat(nodes, dim=0)
        edges = torch.cat(edges, dim=0)

        question_edge = self.question_edge_proj(question_embed)
        question_node = self.question_node_proj(question_embed)

        question_edges = question_edge + edges
        question_nodes = question_node + nodes

        q_edges = self.edge_prj(question_edges)

        q = torch.cat([q_edges, question_nodes], dim=1)
        mid = self.mid_prj(q)
        pred = self.pre_prj(mid)

        return pred