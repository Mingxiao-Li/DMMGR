import torch
import os
import json
import torch.nn as nn
import dgl
import pdb
import torch.nn.functional as F
from model_zoo.graph_models.graph_model_zoo.graph_modules import LanguageEncoder
from model_zoo.graph_models.graph_model_zoo.graph_units import (
    ProbabilityGraph,
    GCN,
    LangGAT,
)
from model_zoo.graph_models.graph_model_zoo.net_utils import (
    cosine,
    generate_bidirection_edges,
)
from x.core.registry import registry
from x.common.util import get_numpy_word_embed


@registry.register_model(name="GATLangGraph")
class GATLangGraph(nn.Module):
    # word-level attention networks
    def __init__(self, config):
        super(GATLangGraph, self).__init__()
        self.config = config
        self.max_lang_edges = config.max_lang_edges
        self.num_layers = config.num_layers
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
        self.lang_encoder = LanguageEncoder(config.LANG_ENCODER)

        self.graph_modules = nn.ModuleList()

        for _ in range(config.num_layers):
            lang_gat_layer = LangGAT(config.LANG_GRAPH)
            graph_gat_layer = GCN(config.GCN)
            cross_att_layer = CrossAtten(config.CROSS_ATT)
            pro_graph_layer = ProbabilityGraph()

            self.graph_modules.append(
                GraphPropageLayer(
                    lang_layer=lang_gat_layer,
                    prob_layer=pro_graph_layer,
                    graph_layer=graph_gat_layer,
                    cross_layer=cross_att_layer,
                )
            )

        self.lang_prj = nn.Linear(config.embeded_size, config.hidden_size)
        self.node_prj = nn.Linear(config.embeded_size, config.hidden_size)
        self.edge_prj = nn.Linear(config.embeded_size, config.hidden_size)
        self.pred_head = PredictionHead(config.PRED_HEAD)

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
        lang_node1_ids_list = item["lang_node1_ids_list"]
        lang_node2_ids_list = item["lang_node2_ids_list"]

        batch_size, _ = question.shape
        device = question.device

        # build lang graph
        word_emb_seq = self.word_embedding(question)
        num_nodes = torch.sum(question_mask != 1, dim=1)  # check
        num_edges = torch.sum(lang_node1_ids_list != 100, dim=1)
        seq_len = num_nodes
        _, _, (ques_emb, _) = self.lang_encoder(word_emb_seq, seq_len)

        lang_graphs = []
        for b in range(batch_size):
            num_node = num_nodes[b].item()
            num_edge = num_edges[b].item()
            g = dgl.graph(
                (
                    lang_node1_ids_list[b][:num_edge].cpu(),
                    lang_node2_ids_list[b][:num_edge].cpu(),
                ),
                num_nodes=num_node,
            )
            g = dgl.to_bidirected(g)
            g = g.to(device)
            g.ndata["feat"] = self.lang_prj(word_emb_seq[b][:num_node])
            lang_graphs.append(g)
        lang_graph_batch = dgl.batch(lang_graphs)

        # build graph and merge
        # get kg embedding
        num_kg_nodes = torch.sum(torch.sum(kg_nodes, dim=2) != self.max_kg_len, dim=1)
        num_kg_edges = torch.sum(torch.sum(kg_edges, dim=2) != self.max_kg_len, dim=1)

        kg_node_rel_len = torch.sum(kg_nodes != 1, dim=2)
        kg_node_rel_len += kg_node_rel_len == 0
        kg_edge_rel_len = torch.sum(kg_edges != 1, dim=2)
        kg_edge_rel_len += kg_edge_rel_len == 0

        kg_node_embedding = self.word_embedding(kg_nodes)
        kg_edge_embedding = self.word_embedding(kg_edges)

        kg_node_feat = torch.sum(kg_node_embedding, dim=2)
        kg_node_feat = kg_node_feat / (
            kg_node_rel_len.unsqueeze(2).expand_as(kg_node_feat)
        )

        kg_edge_feat = torch.sum(kg_edge_embedding, dim=2)
        kg_edge_feat = kg_edge_feat / (
            kg_edge_rel_len.unsqueeze(2).expand_as(kg_edge_feat)
        )

        # get img
        num_img_nodes = torch.sum(
            torch.sum(img_nodes, dim=2) != self.max_img_node_len, dim=1
        )
        num_img_edges = torch.sum(
            torch.sum(img_edges, dim=2) != self.max_img_edge_len, dim=1
        )

        img_node_rel_len = torch.sum(img_nodes != 1, dim=2)
        img_node_rel_len += img_node_rel_len == 0
        img_edge_rel_len = torch.sum(img_edges != 1, dim=2)
        img_edge_rel_len += img_edge_rel_len == 0

        img_node_embedding = self.word_embedding(img_nodes)
        img_edge_embedding = self.word_embedding(img_edges)

        img_node_embedding = torch.sum(img_node_embedding, dim=2)
        img_edge_embedding = torch.sum(img_edge_embedding, dim=2)

        img_node_feat = img_node_embedding / img_node_rel_len.unsqueeze(2).expand_as(
            img_node_embedding
        )
        img_edge_feat = img_edge_embedding / img_edge_rel_len.unsqueeze(2).expand_as(
            img_edge_embedding
        )

        graphs = []
        probability_graphs = []
        for b in range(batch_size):
            num_img_node = num_img_nodes[b].item()
            num_img_edge = num_img_edges[b].item()
            num_kg_node = num_kg_nodes[b].item()
            num_kg_edge = num_kg_edges[b].item()

            img_node1_id_list = img_node1_ids_list[b][:num_img_edge]
            img_node2_id_list = img_node2_ids_list[b][:num_img_edge]
            kg_node1_id_list = kg_node1_ids_list[b][:num_kg_edge]
            kg_node2_id_list = kg_node2_ids_list[b][:num_kg_edge]

            cosine_sim = cosine(
                img_node_feat[b][:num_img_node], kg_node_feat[b][:num_kg_node]
            )

            pos = torch.argmax(cosine_sim).item()
            img_id, kg_id = (
                pos // num_kg_node,
                pos - (num_kg_node * (pos // num_kg_node)),
            )

            all_nodes = torch.cat(
                [img_node_feat[b][:num_img_node], kg_node_feat[b][:num_kg_node]], dim=0
            )
            all_edges = torch.cat(
                [img_edge_feat[b][:num_img_edge], kg_edge_feat[b][:num_kg_edge]], dim=0
            )
            kg_node1_id_list += num_img_node
            kg_node2_id_list += num_img_node

            # merge node index
            for i in range(num_kg_edge):
                if kg_node2_id_list[i].item() == kg_id + num_img_node:
                    kg_node2_id_list[i] = img_id
                if kg_node1_id_list[i].item() == kg_id + num_img_node:
                    kg_node1_id_list[i] = img_id

            all_node1_id_list = torch.cat([img_node1_id_list, kg_node1_id_list], dim=0)
            all_node2_id_list = torch.cat([img_node2_id_list, kg_node2_id_list], dim=0)
            g = dgl.graph(
                (all_node1_id_list, all_node2_id_list),
                num_nodes=num_img_node + num_kg_node,
            )
            g.ndata["feat"] = self.node_prj(all_nodes)
            g.edata["feat"] = self.edge_prj(all_edges)
            # g.remove_nodes(torch.tensor([kg_id + num_img_node], device=device))
            graphs.append(g)

            node1_list, node2_list = generate_bidirection_edges(
                all_node1_id_list, all_node2_id_list
            )
            p_g = dgl.graph(
                (torch.tensor(node1_list).cpu(), torch.tensor(node2_list).cpu()),
                num_nodes=num_img_node + num_kg_node,
            )

            # p_g.remove_nodes(torch.tensor([kg_id + num_img_node]))

            p_g.ndata["att"] = torch.ones(num_img_node + num_kg_node)
            p_g.edata["att"] = torch.ones(p_g.num_edges())
            p_g = p_g.to(device)
            probability_graphs.append(p_g)

        graph_batch = dgl.batch(graphs)
        proba_graph_batch = dgl.batch(probability_graphs)

        for graph_module in self.graph_modules:
            lang_graph_batch, graph_batch = graph_module(
                lang_graph_batch, proba_graph_batch, graph_batch
            )

        pred = self.pred_head(graph_batch, ques_emb)

        return pred


8


class CrossAtten(nn.Module):
    def __init__(self, config):
        super(CrossAtten, self).__init__()
        # word level self attention first
        self.self_att_prj = nn.Linear(config.in_dim, config.hidden_size)
        self.att_prj_value = nn.Linear(config.hidden_size, 1)

        # lang-visual attention
        self.lang_node_proj = nn.Linear(config.hidden_size, config.att_node_size)
        self.lang_edge_proj = nn.Linear(config.hidden_size, config.att_edge_size)

        self.graph_node_proj = nn.Linear(config.node_size, config.att_node_size)

        self.graph_edge_proj = nn.Linear(config.edge_size, config.att_edge_size)

        self.graph_node_att_value_proj = nn.Linear(config.att_node_size, 1)

        self.graph_edge_att_value_proj = nn.Linear(config.att_edge_size, 1)

    def forward(self, lang_graph, graph, prob_graph):
        lang_graphs = dgl.unbatch(lang_graph)
        prob_graphs = dgl.unbatch(prob_graph)
        graphs = dgl.unbatch(graph)

        assert len(lang_graphs) == len(graphs)
        batch_size = len(dgl.unbatch(lang_graph))

        for b in range(batch_size):
            lang_graph = lang_graphs[b]
            graph = graphs[b]

            lang = lang_graph.ndata["feat"]
            graph_nodes = graph.ndata["feat"]
            graph_edges = graph.edata["feat"]

            att_lang = torch.tanh(self.self_att_prj(lang))
            att_value = F.softmax(self.att_prj_value(att_lang).squeeze(1), dim=0)
            lang = torch.matmul(att_value.unsqueeze(0), lang)

            lang_node_att = self.lang_node_proj(lang)
            lang_edge_att = self.lang_edge_proj(lang)
            graph_node_att = self.graph_node_proj(graph_nodes)
            graph_edge_att = self.graph_edge_proj(graph_edges)

            node_att_prj = torch.tanh(lang_node_att + graph_node_att)
            node_att_value = self.graph_node_att_value_proj(node_att_prj)
            node_att_value = F.softmax(node_att_value.squeeze(1), dim=0)

            edge_att_prj = torch.tanh(lang_edge_att + graph_edge_att)
            edge_att_value = self.graph_edge_att_value_proj(edge_att_prj)
            edge_att_value = F.softmax(edge_att_value.squeeze(1), dim=0)

            prob_graphs[b].ndata["att"] = node_att_value
            prob_graphs[b].edata["att"] = edge_att_value.repeat(2)
        prob_graph_batch = dgl.batch(prob_graphs)
        return prob_graph_batch


class GraphPropageLayer(nn.Module):
    def __init__(self, lang_layer, prob_layer, graph_layer, cross_layer):
        super(GraphPropageLayer, self).__init__()
        self.lang_graph_layer = lang_layer
        self.graph_layer = graph_layer
        self.cross_layer = cross_layer
        self.prob_layer = prob_layer

    def forward(self, lang_graph, prob_graph, graph):
        batch_size = len(dgl.unbatch(graph))
        lang_graph = self.lang_graph_layer(lang_graph)
        prob_graph = self.cross_layer(lang_graph, graph, prob_graph)
        prob_graph = self.prob_layer(prob_graph)
        # unbatch
        graphs = dgl.unbatch(graph)
        prob_graphs = dgl.unbatch(prob_graph)
        for b in range(batch_size):
            graphs[b].ndata["att"] = F.softmax(prob_graphs[b].ndata["att"])
            att_value = prob_graphs[b].edata["att"]
            graphs[b].edata["att"] = F.softmax(att_value[: int(len(att_value) / 2)])

        graph = dgl.batch(graphs)
        graph = self.graph_layer(graph)

        return lang_graph, graph


class PredictionHead(nn.Module):
    def __init__(self, config):
        super(PredictionHead, self).__init__()
        self.lang_fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.mid_fc = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.mid_fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.answer_fc = nn.Linear(config.hidden_size, config.num_answers)

    def forward(self, graph, question_emb):
        nodes_out = []
        edges_out = []
        graphs = dgl.unbatch(graph)
        for g in graphs:
            node_out = g.ndata["feat"]
            edge_out = g.edata["feat"]
            node_out = torch.sum(node_out, dim=0)
            edge_out = torch.sum(edge_out, dim=0)
            nodes_out.append(node_out)
            edges_out.append(edge_out)
        nodes_out = torch.stack(nodes_out)
        edges_out = torch.stack(edges_out)

        g_out = torch.cat([nodes_out, edges_out], dim=1)
        lang = self.lang_fc(question_emb)
        lang_g = torch.cat([g_out, lang], dim=1)
        mid_re = F.relu(self.mid_fc2(F.relu(self.mid_fc(lang_g))))
        pred = self.answer_fc(mid_re)

        return pred
