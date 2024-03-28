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


@registry.register_model(name="SimpleGAT")
class SimpleGAT(nn.Module):
    def __init__(self, config):
        super(SimpleGAT, self).__init__()
        self.config = config
        self.max_img_node_len = config.max_img_node_len
        self.max_img_edge_len = config.max_img_edge_len
        self.max_kg_len = config.max_kg_len
        self.use_scene_graph = config.use_scene_graph
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

        if not self.use_scene_graph:
            self.img_node_proj = nn.Linear(config.img_feat_size, config.embeded_size)
            self.img_edge_proj = nn.Linear(config.img_loc_size, config.embeded_size)

        self.question_node_proj = nn.Linear(
            config.LanguageEncoder.hidden_size, config.out_dim
        )
        self.question_edge_proj = nn.Linear(
            config.LanguageEncoder.hidden_size, config.embeded_size
        )
        self.edge_prj = nn.Linear(config.embeded_size, config.out_dim)

        self.mid_prj = nn.Linear(2 * config.out_dim, 3 * config.out_dim)
        self.pre_prj = nn.Linear(3 * config.out_dim, config.num_answer)

        self.img_gat = GAT(
            in_dim=config.in_dim,
            rel_dim=config.in_dim,
            out_dim=config.hidden_dim,
            num_heads=config.num_heads,
            layer_type=config.graph_layer_type,
        )
        self.kg_gat = GAT(
            in_dim=config.in_dim,
            rel_dim=config.in_dim,
            out_dim=config.hidden_dim,
            num_heads=config.num_heads,
            layer_type=config.graph_layer_type,
        )

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

        # build graphs
        # build kg graph
        batch_size, _, _ = kg_nodes.shape
        num_entites = torch.sum(torch.sum(kg_nodes, dim=2) != self.max_kg_len, dim=1)
        num_edges = torch.sum(torch.sum(kg_edges, dim=2) != self.max_kg_len, dim=1)

        entity_rel_len = torch.sum(kg_nodes != 1, dim=2)
        entity_rel_len += entity_rel_len == 0
        edge_rel_len = torch.sum(kg_edges != 1, dim=2)
        edge_rel_len += edge_rel_len == 0

        entity_embedding = self.word_embedding(kg_nodes)
        edge_embedding = self.word_embedding(kg_edges)
        if self.config.use_lang_encoder:
            entity_feat = self.language_encoder(entity_embedding)
            edge_feat = self.language_encoder(edge_embedding)
        else:
            # use average embedding
            entity_feat = torch.sum(entity_embedding, dim=2)  # question ==>
            entity_feat = entity_feat / (
                entity_rel_len.unsqueeze(2).expand_as(entity_feat)
            )
            edge_feat = torch.sum(edge_embedding, dim=2)
            edge_feat = edge_feat / (edge_rel_len.unsqueeze(2).expand_as(edge_feat))

        kg_graphs = []
        for b in range(batch_size):
            num_entity = num_entites[b].item()
            num_edge = num_edges[b].item()

            g = dgl.graph(
                (kg_node1_ids_list[b][:num_edge], kg_node2_ids_list[b][:num_edge]),
                num_nodes=num_entity,
            )
            g = g.to(device)
            g.ndata["feat"] = entity_feat[b][:num_entity]
            g.edata["feat"] = edge_feat[b][:num_edge]
            kg_graphs.append(g)
        kg_graph_batch = dgl.batch(kg_graphs)

        # build img graph
        if self.use_scene_graph:
            num_nodes = torch.sum(
                torch.sum(img_nodes, dim=2) != self.max_img_node_len, dim=1
            )
            num_edges = torch.sum(
                torch.sum(img_edges, dim=2) != self.max_img_edge_len, dim=1
            )
            node_rel_len = torch.sum(img_nodes != 1, dim=2)
            node_rel_len += node_rel_len == 0
            edge_rel_len = torch.sum(img_edges != 1, dim=2)
            edge_rel_len += edge_rel_len == 0
            node_embedding = self.word_embedding(img_nodes)
            edge_embedding = self.word_embedding(img_edges)
            node_embedding = torch.sum(node_embedding, dim=2)
            edge_embedding = torch.sum(edge_embedding, dim=2)

            node_embedding = node_embedding / node_rel_len.unsqueeze(2).expand_as(
                node_embedding
            )
            edge_embedding = edge_embedding / edge_rel_len.unsqueeze(2).expand_as(
                edge_embedding
            )
            img_graphs = []
            for b in range(batch_size):
                num_node = num_nodes[b].item()
                num_edge = num_edges[b].item()
                g = dgl.graph(
                    (
                        img_node1_ids_list[b][:num_edge],
                        img_node2_ids_list[b][:num_edge],
                    ),
                    num_nodes=num_node,
                )
                g = g.to(device)
                g.ndata["feat"] = node_embedding[b][:num_node]
                g.edata["feat"] = edge_embedding[b][:num_edge]
                img_graphs.append(g)

        else:
            batch_size, num_obj, _ = img_nodes.shape
            img_node_feat = self.img_node_proj(img_nodes)
            img_edge_feat = self.img_edge_proj(img_edges)
            img_graphs = []

            for b in range(batch_size):
                g = dgl.graph(
                    (img_node1_ids_list[b], img_node2_ids_list[b]), num_nodes=num_obj
                )
                g = g.to(device)
                g.ndata["feat"] = img_node_feat[b]
                g.edata["feat"] = img_edge_feat[b]
                img_graphs.append(g)

        img_graph_batch = dgl.batch(img_graphs)

        kg_graph_batch = self.kg_gat(kg_graph_batch)
        img_graph_batch = self.img_gat(img_graph_batch)
        img_graphs = dgl.unbatch(img_graph_batch)
        kg_graphs = dgl.unbatch(kg_graph_batch)
        img_edges = []
        kg_edges = []
        img_nodes = []
        kg_nodes = []
        for i in range(batch_size):
            # pdb.set_trace()
            img_nodes.append(torch.sum(img_graphs[i].ndata["feat"], dim=0).unsqueeze(0))
            kg_nodes.append(torch.sum(kg_graphs[i].ndata["feat"], dim=0).unsqueeze(0))

            img_edges.append(torch.sum(img_graphs[i].edata["feat"], dim=0).unsqueeze(0))
            kg_edges.append(torch.sum(img_graphs[i].edata["feat"], dim=0).unsqueeze(0))
        img_nodes = torch.cat(img_nodes, dim=0)
        img_edges = torch.cat(img_edges, dim=0)
        kg_nodes = torch.cat(kg_nodes, dim=0)
        kg_edges = torch.cat(kg_edges, dim=0)

        question_edge = self.question_edge_proj(question_embed)
        question_node = self.question_node_proj(question_embed)

        q_img_edge = question_edge + img_edges  # 1024
        q_kg_edge = question_edge + kg_edges  # 300
        q_img_node = question_node + img_nodes  # 300
        q_kg_node = question_node + kg_nodes  # 1024

        q_kg_edge = self.edge_prj(q_kg_edge)
        q_img_edge = self.edge_prj(q_img_edge)

        q_img = torch.cat([q_img_edge, q_img_node], dim=1)
        q_kg = torch.cat([q_kg_edge, q_kg_node], dim=1)

        mid = self.mid_prj(q_img + q_kg)
        pred = self.pre_prj(mid)
        if torch.any(torch.isnan(pred)):
            pdb.set_trace()
        # print(pred)
        return pred
