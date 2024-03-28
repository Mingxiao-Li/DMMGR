import torch
import os
import json
from torch.functional import norm
import torch.nn as nn
import pdb
import dgl
import torch.nn.functional as F
from torch.nn.functional import normalize
from model_zoo.graph_models.graph_model_zoo.graph_modules import LanguageEncoder
from model_zoo.graph_models.graph_model_zoo.graph_units import ImgGCN, FactGCN
from model_zoo.graph_models.graph_model_zoo.net_utils import *
from x.core.registry import registry
from x.common.util import get_numpy_word_embed
from x.modules.attention import CrossAttention




@registry.register_model(name="GATQuestionGuidedCross")
class GATQuestionGuidedCross(nn.Module):
    def __init__(self, config):
        super(GATQuestionGuidedCross, self).__init__()
        self.config = config
        self.num_layers = config.num_layers
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

        self.dropout = nn.Dropout(config.dropout)

        self.language_encoder = LanguageEncoder(config.LanguageEncoder)
        self.build_txt_command_linear()

        if not self.use_scene_graph:
            self.edge_linear = nn.Linear(config.img_edge_dim, config.hidden_size // 2)
        
        self.img_node_linear = nn.Linear(config.embeded_size, config.hidden_size // 2)
        self.kg_node_linear = nn.Linear(config.embeded_size, config.hidden_size // 2)
        self.kg_edge_linear = nn.Linear(config.embeded_size, config.hidden_size // 2)
        self.final_ques_linear = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.ques_type_mid = nn.Linear(config.hidden_size, config.hidden_size // 2)
        self.ques_type_linear = nn.Linear(config.hidden_size // 2, 2)

        self.gqc = nn.ModuleList()
        for _ in range(config.num_layers):
            self.gqc.append(GATQGCLayer(config.GatQgcLayer))

        self.prediction_head = PredictionHead(config.PredictionHead)

    def build_txt_command_linear(self):
        self.ques_vec_prj = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        for t in range(self.config.num_layers + 1):
            ques_layer2 = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            setattr(self, "ques_layer%d" % t, ques_layer2)
        self.ques_layer_final = nn.Linear(self.config.hidden_size, 1)

    def extract_txt_command(self, ques_vec, ques_ctx, question_mask, t):
        ques_layer2 = getattr(self, "ques_layer%d" % t)
        act_fun = activations[self.config.act_fn]
        q_cmd = ques_layer2(act_fun(self.ques_vec_prj(ques_vec)))
        raw_att = self.ques_layer_final(q_cmd[:, None, :] * ques_ctx).squeeze(-1)
        raw_att = raw_att.masked_fill_(question_mask == 1, -1e32)
        att = F.softmax(raw_att, dim=-1)
        cmd = torch.bmm(att[:, None, :], ques_ctx).squeeze(1)
        return cmd

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
        batch_size, _ = question.shape
        seq_word_emb = self.word_embedding(question)
        seq_len = torch.sum(question_mask == 1, dim=1)
        ques_ctx, _, (ques_vec, _) = self.language_encoder(seq_word_emb, seq_len)
        ques_vec = ques_vec.reshape(batch_size, -1)
        ques_vec = self.dropout(ques_vec)  # batch_size, 1024

        img_node_feat, img_edge_feat, num_img_node, num_img_edge = get_graph_feat(
            img_nodes,
            img_edges,
            self.word_embedding,
            self.max_img_node_len,
            self.max_img_edge_len,
            self.use_scene_graph,
        )
        if not self.config.use_scene_graph:
            img_edge_feat_ = self.edge_linear(img_edge_feat)
            img_edge_feat = normalize(img_edge_feat_)
        img_node_feat = self.img_node_linear(img_node_feat)

        img_graph_batch = build_graph(
            batch_size=batch_size,
            num_nodes=num_img_node,
            num_edges=num_img_edge,
            node_feat=img_node_feat,
            edge_feat=img_edge_feat,
            node1_id_list=img_node1_ids_list,
            node2_id_list=img_node2_ids_list,
            device=device,
        )

        kg_node_feat, kg_edge_feat, num_kg_node, num_kg_edge = get_graph_feat(
            kg_nodes,
            kg_edges,
            self.word_embedding,
            self.max_kg_len,
            self.max_kg_len,
            True,
        )
        kg_node_feat = self.kg_node_linear(kg_node_feat)
        kg_edge_feat = self.kg_edge_linear(kg_edge_feat)

        kg_graph_batch = build_graph(
            batch_size=batch_size,
            num_nodes=num_kg_node,
            num_edges=num_kg_edge,
            node_feat=kg_node_feat,
            edge_feat=kg_edge_feat,
            node1_id_list=kg_node1_ids_list,
            node2_id_list=kg_node2_ids_list,
            device=device,
        )

        for i in range(self.num_layers):
            ques_cmd = self.extract_txt_command(ques_vec, ques_ctx, question_mask, i)
            img_graph_batch, kg_graph_batch = self.gqc[i](
                ques_cmd, img_graph_batch, kg_graph_batch
            )

        # ques_final = self.final_ques_linear(ques_vec)

        ques_final = self.extract_txt_command(
            ques_vec, ques_ctx, question_mask, self.config.num_layers
        )
        # ques_type_mid = self.ques_type_mid(ques_type)
        # question_type_pred = self.ques_type_linear(ques_type_mid)
        # q_r_type_mask = question_type_pred.squeeze(1)[:, 0].unsqueeze(1)
        # q_e_type_mask = question_type_pred.squeeze(1)[:, 1].unsqueeze(1)
        prediction = self.prediction_head(ques_final, img_graph_batch, kg_graph_batch)
        # prediction = torch.cat(
        #    [
        #        q_r_type_mask * prediction[:, :1709],
        #        q_e_type_mask * prediction[:, 1709:],
        #    ],
        #    dim=1,
        # )
        return prediction  # question_type_pred


class GATQGCLayer(nn.Module):
    def __init__(self, config):
        super(GATQGCLayer, self).__init__()
        self.config = config
        self.qg_img_attention = QuestionGuidedGraphAttention(config.QuestionGuideAttImg)
        self.qg_kg_attention = QuestionGuidedGraphAttention(config.QuestionGuideAttKg)
        self.gnn = GraphNeuralNet(config.in_dims, config.rel_dims, config.out_dims)
        self.cross_att = CrossGraphAttention(config.CrossGraphAtt)

    def forward(
        self,
        question,
        img_graphs_batch,
        kg_graphs_batch,
    ):

        batch_size, _ = question.shape
        img_graphs_batch_ori = img_graphs_batch
        kg_graphs_batch_ori = kg_graphs_batch

        img_att_graphs_batch = self.qg_img_attention(question, img_graphs_batch)
        kg_att_graphs_batch = self.qg_kg_attention(question, kg_graphs_batch)

        img_graphs_batch, kg_graphs_batch = self.gnn(
            img_att_graphs_batch, kg_att_graphs_batch
        )
        img_graphs_batch, kg_graphs_batch = self.cross_att(
            question, img_graphs_batch, kg_graphs_batch
        )

        img_graphs = dgl.unbatch(img_graphs_batch)
        kg_graphs = dgl.unbatch(kg_graphs_batch)
        img_graphs_ori = dgl.unbatch(img_graphs_batch_ori)
        kg_graphs_ori = dgl.unbatch(kg_graphs_batch_ori)

        for i in range(batch_size):
            img_graphs[i].ndata["feat"] = (
                img_graphs_ori[i].ndata["feat"] + img_graphs[i].ndata["feat"]
            )
            kg_graphs[i].ndata["feat"] = (
                kg_graphs_ori[i].ndata["feat"] + kg_graphs[i].ndata["feat"]
            )

            img_graphs[i].ndata["feat"] = normalize(img_graphs[i].ndata["feat"])
            kg_graphs[i].ndata["feat"] = normalize(kg_graphs[i].ndata["feat"])
        img_graphs_batch = dgl.batch(img_graphs)
        kg_graphs_batch = dgl.batch(kg_graphs)
        return img_graphs_batch, kg_graphs_batch


class QuestionGuidedGraphAttention(nn.Module):
    def __init__(self, config) -> None:
        super(QuestionGuidedGraphAttention, self).__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        # question guided node attention
        self.node_att_prj_ques = nn.Linear(config.ques_dim, config.node_att_prj_dim)
        self.node_att_prj = nn.Linear(config.node_dim, config.node_att_prj_dim)
        self.node_att_value = nn.Linear(config.node_att_prj_dim, 1)

        # question guided edge attention
        self.edge_att_prj_ques = nn.Linear(config.ques_dim, config.edge_att_prj_dim)
        self.edge_att_prj = nn.Linear(config.edge_dim, config.edge_att_prj_dim)
        self.edge_att_value = nn.Linear(config.edge_att_prj_dim, 1)

    def forward(self, question, graphs_batch):

        batch_size, _ = question.shape
        graphs = dgl.unbatch(graphs_batch)
        question = self.dropout(question)
        ques_node_att_prj_embeds = self.node_att_prj_ques(question)
        ques_edge_att_prj_embeds = self.edge_att_prj_ques(question)

        for i in range(batch_size):
            graph = graphs[i]
            nodes = graph.ndata["feat"]
            edges = graph.edata["feat"]

            # question node att
            node_att_prj_embed = self.node_att_prj(nodes)

            ques_node_att_prj_embed = (
                ques_node_att_prj_embeds[i, :]
                .unsqueeze(0)
                .repeat(node_att_prj_embed.shape[0], 1)
            )
            node_att_value = self.node_att_value(
                torch.tanh(ques_node_att_prj_embed + node_att_prj_embed)
            )
            node_att_value = F.softmax(node_att_value, dim=1)

            # question edge att
            edge_att_prj_embed = self.edge_att_prj(edges)

            ques_edge_att_prj_embed = (
                ques_edge_att_prj_embeds[i, :]
                .unsqueeze(0)
                .repeat(edge_att_prj_embed.shape[0], 1)
            )

            edge_att_value = self.edge_att_value(
                torch.tanh(ques_edge_att_prj_embed + edge_att_prj_embed)
            )
            edge_att_value = F.softmax(edge_att_value, dim=1)
            graph.ndata["att"] = node_att_value.squeeze(1)
            graph.edata["att"] = edge_att_value.squeeze(1)

        graphs_batch = dgl.batch(graphs)

        return graphs_batch


class GraphNeuralNet(nn.Module):
    def __init__(self, in_dims, rel_dims, out_dims):
        super(GraphNeuralNet, self).__init__()

        self.img_gcn = ImgGCN(in_dims=in_dims, rel_dims=rel_dims, out_dims=out_dims)
        self.kg_gcn = FactGCN(in_dims=in_dims, rel_dims=rel_dims, out_dims=out_dims)

    def forward(self, img_graph, kg_graph):

        img_graph = self.img_gcn(img_graph)
        kg_graph = self.kg_gcn(kg_graph)

        return img_graph, kg_graph


class CrossGraphAttention(nn.Module):
    def __init__(self, config):
        super(CrossGraphAttention, self).__init__()
        self.config = config
        self.drop_out = nn.Dropout(config.dropout)
        self.ques_img_prj = nn.Linear(config.ques_dim, config.img_node_dim)
        self.ques_kg_prj = nn.Linear(config.ques_dim, config.kg_node_dim)

        self.img_to_kg_multi_head_cross_attention = CrossAttention(
            hidden_size=config.hidden_size,
            multi_head=config.num_heads,
            hidden_size_head=config.hidden_size_head,
            dropout_r=config.dropout,
            mid_size=config.mid_size,
            act_fn="relu",
            use_ffn=True,
        )

        self.kg_to_img_multi_head_cross_attention = CrossAttention(
            hidden_size=config.hidden_size,
            multi_head=config.num_heads,
            hidden_size_head=config.hidden_size_head,
            dropout_r=config.dropout,
            mid_size=config.mid_size,
            act_fn="relu",
            use_ffn=True,
        )

    def forward(self, question, img_graphs_batch, kg_graphs_batch):

        batch_size, _ = question.shape
        img_graphs = dgl.unbatch(img_graphs_batch)
        kg_graphs = dgl.unbatch(kg_graphs_batch)
        question = self.drop_out(question)

        question_img = self.ques_img_prj(question)
        question_kg = self.ques_kg_prj(question)
        for i in range(batch_size):
            img_graph_nodes = img_graphs[i].ndata["feat"]
            kg_graph_nodes = kg_graphs[i].ndata["feat"]
            question_img_emb = (
                question_img[i, :].unsqueeze(0).repeat(img_graph_nodes.shape[0], 1)
            )
            question_kg_emb = (
                question_kg[i, :].unsqueeze(0).repeat(kg_graph_nodes.shape[0], 1)
            )

            img_node_feat = torch.tanh(question_img_emb + img_graph_nodes).unsqueeze(0)

            kg_node_feat = torch.tanh(question_kg_emb + kg_graph_nodes).unsqueeze(0)

            img_node_output_feat = self.img_to_kg_multi_head_cross_attention(
                kg_node_feat,
                kg_node_feat,
                img_node_feat,
                None,
            )

            kg_node_output_feat = self.kg_to_img_multi_head_cross_attention(
                img_node_feat,
                img_node_feat,
                kg_node_feat,
                None,
            )
            img_graphs[i].ndata["feat"] = img_node_output_feat.squeeze(0)
            kg_graphs[i].ndata["feat"] = kg_node_output_feat.squeeze(0)
        img_graphs_batch = dgl.batch(img_graphs)
        kg_graphs_batch = dgl.batch(kg_graphs)

        return img_graphs_batch, kg_graphs_batch


class PredictionHead(nn.Module):
    def __init__(self, config):
        super(PredictionHead, self).__init__()
        self.config = config

        self.question_edge_prj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.question_node_prj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.edge_prj = nn.Linear(config.hidden_size, config.hidden_size)
        self.pred_head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size*2, config.mid_size),
            nn.ELU(),
            nn.Linear(config.mid_size, config.num_answers),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, question, img_graph_batch, kg_graph_batch):

        batch_size, _ = question.shape
        img_graphs = dgl.unbatch(img_graph_batch)
        kg_graphs = dgl.unbatch(kg_graph_batch)
        img_nodes = []
        kg_nodes = []
        img_edges = []
        kg_edges = []
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
        kg_nodes = torch.cat(kg_nodes, dim=0)
        img_edges = torch.cat(img_edges, dim=0)
        kg_edges = torch.cat(kg_edges, dim=0)

        question_edge = self.question_edge_prj(question)
        question_node = self.question_node_prj(question)

        q_img_edge = question_edge + img_edges
        q_kg_edge = question_edge + kg_edges

        q_img_node = question_node + img_nodes
        q_kg_node = question_node + kg_nodes

        q_img = torch.cat([q_img_edge, q_img_node], dim=1)
        q_kg = torch.cat([q_kg_edge, q_kg_node], dim=1)

        pred = self.pred_head(q_img + q_kg)

        return pred
