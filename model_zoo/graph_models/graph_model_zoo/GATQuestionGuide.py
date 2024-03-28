import torch
import os
import json
import torch.nn as nn
import dgl
import pdb
import torch.nn.functional as F
from model_zoo.graph_models.graph_model_zoo.graph_units import ImgGCN, FactGCN
from model_zoo.graph_models.graph_model_zoo.graph_modules import LanguageEncoder
from x.core.registry import registry
from x.common.util import get_numpy_word_embed
from torch.nn.functional import normalize
from model_zoo.graph_models.graph_model_zoo.net_utils import *


@registry.register_model(name="GATQuestionGuide")
class GATQuestionGuide(nn.Module):
    def __init__(self, config):
        super(GATQuestionGuide, self).__init__()
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

        self.question_node_proj = nn.Linear(
            config.LanguageEncoder.hidden_size, config.out_dim
        )
        self.question_edge_proj = nn.Linear(
            config.LanguageEncoder.hidden_size, config.embeded_size
        )

        if not self.config.use_scene_graph:
            self.edge_linear = nn.Linear(config.img_edge_dim, config.embeded_size)

        self.edge_prj = nn.Linear(config.embeded_size, config.out_dim)

        if self.config.load_answer_embeddings:
            candidate_answer_embedding = torch.load(
                os.path.join(config.parent_path, config.candidate_answer_emb_path)
            )
            self.mid_prj = nn.Linear(2 * config.out_dim, config.embeded_size)
            self.pre_prj = nn.Linear(config.embeded_size, config.embeded_size)
            self.out_prj = nn.Linear(config.embeded_size, config.num_answer)
            self.out_prj.weight.date = candidate_answer_embedding
            self.out_prj.weight.requires_grad = False
        else:
            self.mid_prj = nn.Linear(2 * config.out_dim, config.out_dim)
            self.pre_prj = nn.Linear(config.out_dim, config.num_answer)

        # question guided img node attention
        self.img_node_att_proj_ques = nn.Linear(
            config.hidden_dim, config.img_node_att_ques_img_prj_dim
        )
        self.img_node_att_proj_img = nn.Linear(
            config.img_node_dim, config.img_node_att_ques_img_prj_dim
        )
        self.img_node_att_value = nn.Linear(config.img_node_att_ques_img_prj_dim, 1)

        # question guided img edge attention
        self.img_edge_att_proj_ques = nn.Linear(
            config.hidden_dim, config.img_edge_att_ques_rel_proj_dim
        )
        self.img_edge_att_proj_edge = nn.Linear(
            config.embeded_size, config.img_edge_att_ques_rel_proj_dim
        )
        self.img_edge_att_value = nn.Linear(config.img_edge_att_ques_rel_proj_dim, 1)

        # question guided knowledge node attention
        self.kg_node_att_proj_ques = nn.Linear(
            config.hidden_dim, config.kg_node_att_ques_node_proj_dims
        )
        self.kg_node_att_proj_node = nn.Linear(
            config.embeded_size, config.kg_node_att_ques_node_proj_dims
        )
        self.kg_node_att_value = nn.Linear(config.kg_node_att_ques_node_proj_dims, 1)

        # question guided knowledge edge attention
        self.kg_edge_att_proj_ques = nn.Linear(
            config.hidden_dim, config.kg_edge_att_ques_edge_proj_dims
        )
        self.kg_edge_att_proj_edge = nn.Linear(
            config.embeded_size, config.kg_edge_att_ques_edge_proj_dims
        )
        self.kg_edge_att_value = nn.Linear(config.kg_edge_att_ques_edge_proj_dims, 1)

        ##
        self.img_gat = ImgGCN(
            in_dims=config.in_dim,
            rel_dims=config.in_dim,
            out_dims=config.hidden_dim,
        )
        self.kg_gat = FactGCN(
            in_dims=config.in_dim,
            out_dims=config.hidden_dim,
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

        batch_size, _ = question.shape
        device = question.device
        seq_word_emb = self.word_embedding(question)
        seq_len = torch.sum(question_mask == 1, dim=1)
        all_out, lens, (hidden_out, cell_out) = self.language_encoder(
            seq_word_emb, seq_len
        )
        question_embed = hidden_out

        # question img node att
        img_node_att_proj_ques_embed = self.img_node_att_proj_ques(question_embed)
        img_node_feat, img_edge_feat, num_img_node, num_img_edge = get_graph_feat(
            img_nodes,
            img_edges,
            self.word_embedding,
            self.max_img_node_len,
            self.max_img_edge_len,
            self.config.use_scene_graph,
        )

        if not self.config.use_scene_graph:
            img_edge_feat_ = self.edge_linear(img_edge_feat)
            img_edge_feat = normalize(img_edge_feat_)

        img_node_att_proj_img_embed = self.img_node_att_proj_img(img_node_feat)
        img_node_mask = torch.sum(img_nodes, dim=2) == self.max_img_node_len
        img_node_att_proj_ques_embed = img_node_att_proj_ques_embed.unsqueeze(1).repeat(
            1, img_node_att_proj_img_embed.shape[1], 1
        )
        img_node_att_value = self.img_node_att_value(
            torch.tanh(img_node_att_proj_ques_embed + img_node_att_proj_img_embed)
        )
        img_node_att_value = img_node_att_value.squeeze(2).masked_fill(
            img_node_mask == 1, -1e9
        )
        img_node_att_value = F.softmax(img_node_att_value, dim=1)

        # question img edge att
        img_edge_att_proj_ques_embed = self.img_edge_att_proj_ques(question_embed)
        img_edge_att_proj_img_embed = self.img_edge_att_proj_edge(img_edge_feat)
        img_edge_mask = torch.sum(img_edges, dim=2) == self.max_img_edge_len
        img_edge_att_proj_ques_embed = img_edge_att_proj_ques_embed.unsqueeze(1).repeat(
            1, img_edge_att_proj_img_embed.shape[1], 1
        )
        img_edge_att_value = self.img_edge_att_value(
            torch.tanh(img_edge_att_proj_ques_embed + img_edge_att_proj_img_embed)
        )
        img_edge_att_value = img_edge_att_value.squeeze(2).masked_fill(
            img_edge_mask == 1, -1e9
        )
        img_edge_att_value = F.softmax(img_edge_att_value, dim=1)

        # question kg node att
        kg_node_att_proj_ques_embed = self.kg_node_att_proj_ques(question_embed)
        kg_node_feat, kg_edge_feat, num_kg_node, num_kg_edge = get_graph_feat(
            kg_nodes,
            kg_edges,
            self.word_embedding,
            self.max_kg_len,
            self.max_kg_len,
            True,
        )

        kg_node_att_proj_kg_embed = self.kg_node_att_proj_node(kg_node_feat)
        kg_node_mask = torch.sum(kg_nodes, dim=2) == self.max_kg_len

        kg_node_att_proj_ques_embed = kg_node_att_proj_ques_embed.unsqueeze(1).repeat(
            1, kg_node_att_proj_kg_embed.shape[1], 1
        )
        kg_node_att_value = self.kg_node_att_value(
            torch.tanh(kg_node_att_proj_ques_embed + kg_node_att_proj_kg_embed)
        )
        kg_node_att_value = kg_node_att_value.squeeze(2).masked_fill(
            kg_node_mask == 1, -1e9
        )
        kg_node_att_value = F.softmax(kg_node_att_value, dim=1)

        # question kg edge att
        kg_edge_att_proj_ques_embed = self.kg_edge_att_proj_ques(question_embed)
        kg_edge_att_proj_kg_embed = self.kg_edge_att_proj_edge(kg_edge_feat)
        kg_edge_mask = torch.sum(kg_edges, dim=2) == self.max_kg_len
        kg_edge_att_proj_ques_embed = kg_edge_att_proj_ques_embed.unsqueeze(1).repeat(
            1, kg_edge_att_proj_kg_embed.shape[1], 1
        )
        kg_edge_att_value = self.kg_edge_att_value(
            torch.tanh(kg_edge_att_proj_ques_embed + kg_edge_att_proj_kg_embed)
        )
        kg_edge_att_value = kg_edge_att_value.squeeze(2).masked_fill(
            kg_edge_mask == 1, -1e9
        )
        kg_edge_att_value = F.softmax(kg_edge_att_value, dim=1)

        # build kg grap
        kg_graph_batch = build_graph(
            batch_size=batch_size,
            num_nodes=num_kg_node,
            num_edges=num_kg_edge,
            node_feat=kg_node_feat,
            edge_feat=kg_edge_feat,
            node1_id_list=kg_node1_ids_list,
            node2_id_list=kg_node2_ids_list,
            edge_att_value=kg_edge_att_value,
            node_att_value=kg_node_att_value,
            device=device,
        )

        # build img graph
        img_graph_batch = build_graph(
            batch_size=batch_size,
            num_nodes=num_img_node,
            num_edges=num_img_edge,
            node_feat=img_node_feat,
            edge_feat=img_edge_feat,
            node1_id_list=img_node1_ids_list,
            node2_id_list=img_node2_ids_list,
            edge_att_value=img_edge_att_value,
            node_att_value=img_node_att_value,
            device=device,
        )

        kg_graph_batch = self.kg_gat(kg_graph_batch)
        img_graph_batch = self.img_gat(img_graph_batch)

        img_graphs = dgl.unbatch(img_graph_batch)
        kg_graphs = dgl.unbatch(kg_graph_batch)

        img_edges = []
        kg_edges = []
        img_nodes = []
        kg_nodes = []
        for i in range(batch_size):
            img_nodes.append(
                torch.mean(img_graphs[i].ndata["feat"], dim=0).unsqueeze(0)
            )
            kg_nodes.append(torch.mean(kg_graphs[i].ndata["feat"], dim=0).unsqueeze(0))

            img_edges.append(
                torch.mean(img_graphs[i].edata["feat"], dim=0).unsqueeze(0)
            )
            kg_edges.append(torch.mean(img_graphs[i].edata["feat"], dim=0).unsqueeze(0))
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
        if self.config.load_answer_embeddings:
            pred = self.out_prj(pred)
        # pred = nn.Sigmoid()(pred)
        if torch.any(torch.isnan(pred)):
            pdb.set_trace()

        return pred