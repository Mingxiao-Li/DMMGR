import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from x.core.registry import registry
from x.modules.dynamic_rnn import DynamicRNN
import pdb
import math


@registry.register(type="LanuageEncoder", name="LSTMEncoder")
class LanguageEncoder(nn.Module):
    def __init__(self, config):
        super(LanguageEncoder, self).__init__()
        self.config = config
        if config.rnn_type == "lstm":
            rnn = nn.LSTM(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                bidirectional=config.bidirectional,
                dropout=config.dropout,
                batch_first=True,
            )
        elif config.rnn_type == "gru":
            rnn = nn.GRU(
                input_size=config.input_size,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                bidriectional=config.bidriectional,
                dropout=config.dropout,
                batch_first=True,
            )
        self.rnn = DynamicRNN(rnn, config.output_last_layer)  # set false

    def forward(self, seq_embedding, seq_len):
        output, lens, (h, c) = self.rnn(seq_embedding, seq_len)

        return output, lens, (h, c)


class RGraphAttention(nn.Module):
    def __init__(
        self,
        hidden_size,
        att_fuse_method="add",
    ):
        super(RGraphAttention, self).__init__()
        self.att_fuse_method = att_fuse_method
        self.guild_proj = nn.Linear(hidden_size, hidden_size)
        self.node_proj = nn.Linear(hidden_size, hidden_size)
        self.edge_proj = nn.Linear(hidden_size, hidden_size)
        if self.att_fuse_method == "cat":
            self.node_att_proj = nn.Linear(hidden_size * 4, 1, bias=False)
            self.edge_att_proj = nn.Linear(hidden_size * 3, 1, bias=False)
        elif self.att_fuse_method == "add":
            self.node_att_proj = nn.Linear(hidden_size, 1, bias=False)
            self.edge_att_proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, g):

        g.ndata["feat"] = self.node_proj(g.ndata["feat"])
        g.edata["feat"] = self.edge_proj(g.edata["feat"])
        g.edata["guild_vec"] = self.guild_proj(g.edata["guild_vec"])
        g.apply_edges(self.edge_feat_update)
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g

    def message_func(self, edges):
        return {"src": edges.src["feat"], "score": edges.data["score"]}

    def edge_attention(self, edges):
        z = self.att_fuse(
            (
                [
                    edges.src["feat"],
                    edges.dst["feat"],
                    edges.data["feat"],
                    edges.data["guild_vec"],
                ]
            )
        )
        a = self.node_att_proj(z)
        return {"score": F.leaky_relu(a)}

    def edge_feat_update(self, edges):
        z = torch.cat(
            [edges.src["feat"].unsqueeze(1), edges.dst["feat"].unsqueeze(1)], dim=1
        )
        # shape  total edge, 2 , dim
        z1 = self.edge_att_proj(
            self.att_fuse(
                [edges.src["feat"], edges.data["feat"], edges.data["guild_vec"]]
            )
        )
        z2 = self.edge_att_proj(
            self.att_fuse(
                [edges.dst["feat"], edges.data["feat"], edges.data["guild_vec"]]
            )
        )
        a1 = F.leaky_relu(z1)
        a2 = F.leaky_relu(z2)
        alpha = F.softmax(
            torch.cat([a1, a2], dim=1), dim=1
        )  # totl edge x 2 -> total edge x1 x2
        edge_feat = torch.matmul(alpha.unsqueeze(1), z).squeeze(1) + edges.data["feat"]
        return {"feat": edge_feat}

    def att_fuse(self, ele_list):
        if self.att_fuse_method == "cat":
            return torch.cat(ele_list, dim=1)
        elif self.att_fuse_method == "add":
            ele_list = [e.unsqueeze(0) for e in ele_list]
            return torch.sum(torch.cat(ele_list, dim=0), dim=0)

    def reduce_func(self, nodes):
        belta = F.softmax(nodes.mailbox["score"], dim=1)
        h = torch.sum(belta * nodes.mailbox["src"], dim=1) + nodes.data["feat"]
        return {"feat": h}


class RCrossReason(nn.Module):
    def __init__(self, hidden_size, att_fuse_method):
        super(RCrossReason, self).__init__()
        self.cross_graph_reason = RGraphAttention(
            hidden_size=hidden_size, att_fuse_method=att_fuse_method
        )

    def forward(self, g):
        g = self.cross_graph_reason(g)
        return g


class AttentionAggretate(nn.Module):
    def __init__(self, hidden_size, max_nodes, max_edges):
        super(AttentionAggretate, self).__init__()
        self.hidden_size = hidden_size
        self.max_nodes = max_nodes
        self.max_edges = max_edges

        self.node_proj_k = nn.Linear(hidden_size, hidden_size)
        self.edge_proj_k = nn.Linear(hidden_size, hidden_size)
        self.node_proj_v = nn.Linear(hidden_size, hidden_size)
        self.edge_proj_v = nn.Linear(hidden_size, hidden_size)
        self.ques_proj_q = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def pad_graph_element(self, element, max_num):
        device = element.device
        num_ele, ele_dim = element.shape
        mix_ele_pad = torch.ones(max_num, ele_dim, device=device)
        assert num_ele <= max_num
        mix_ele_pad[:num_ele] = element
        mask = torch.ones(1, max_num, device=device)
        mask[:, :num_ele] = 0
        return mix_ele_pad, mask

    def forward(self, batch_g, ques):
        graph_nodes = []
        graph_edges = []
        nodes_mask = []
        edges_mask = []
        graphs = dgl.unbatch(batch_g)
        for graph in graphs:
            #  some graphs has difference number of nodes and edges
            #  creat pad and mask here
            nodes = graph.ndata["feat"]
            edges = graph.edata["feat"]
            mix_node_pad, node_mask = self.pad_graph_element(nodes, self.max_nodes)
            mix_edge_pad, edge_mask = self.pad_graph_element(edges, self.max_edges)
            nodes_mask.append(node_mask.unsqueeze(0))
            edges_mask.append(edge_mask.unsqueeze(0))
            graph_nodes.append(mix_node_pad.unsqueeze(0))
            graph_edges.append(mix_edge_pad.unsqueeze(0))
            # shape bs num_nodes/edges, hidden_size
        # aggregate

        nodes_mask_tensor = torch.cat(nodes_mask, dim=0)
        edges_mask_tensor = torch.cat(edges_mask, dim=0)
        node_tensor = torch.cat(graph_nodes, dim=0)
        edge_tensor = torch.cat(graph_edges, dim=0)

        nodes_k = self.node_proj_k(node_tensor)
        edges_k = self.edge_proj_k(edge_tensor)
        ques = self.ques_proj_q(ques)
        nodes_v = self.node_proj_v(node_tensor)
        edges_v = self.edge_proj_v(edge_tensor)

        nodes_att_score = torch.matmul(
            ques.unsqueeze(1), nodes_k.permute(0, 2, 1)
        ) / math.sqrt(self.hidden_size)
        edges_att_score = torch.matmul(
            ques.unsqueeze(1), edges_k.permute(0, 2, 1)
        ) / math.sqrt(self.hidden_size)

        nodes_att_score = nodes_att_score.masked_fill(nodes_mask_tensor == 1, -1e32)
        edges_att_score = edges_att_score.masked_fill(edges_mask_tensor == 1, -1e32)

        nodes_att_score = F.softmax(nodes_att_score, dim=2)
        edges_att_score = F.softmax(edges_att_score, dim=2)
        final_node = torch.matmul(nodes_att_score, nodes_v).squeeze(1)
        final_edge = torch.matmul(edges_att_score, edges_v).squeeze(1)
        gate_value = self.sigmoid(F.leaky_relu(self.gate(ques)))
        return final_node * gate_value + (1 - gate_value) * final_edge


class PredictionHead(nn.Module):
    def __init__(self, input_size, mid_size, output_size, dropout_r):
        super(PredictionHead, self).__init__()
        self.dropout_r = dropout_r
        self.proj_1 = nn.Linear(input_size, mid_size)
        self.proj_2 = nn.Linear(mid_size, output_size)
        if self.dropout_r > 0:
            self.dropout_r = nn.Dropout(self.dropout_r)

    def forward(self, x):
        x = self.proj_1(x)
        x = F.leaky_relu(x)
        if self.dropout_r > 0:
            x = self.dropout_r(x)
        x = self.proj_2(x)
        return x
