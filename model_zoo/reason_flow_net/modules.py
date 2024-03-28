import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from x.modules.linear import MLP
from x.modules.dynamic_rnn import DynamicRNN
from typing import Union


class LanguageEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embeded_size: int,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: int,
        rnn_type: str,
        glove_word_emebd: Union[None, np.array] = None,
        bidirectional: bool = False,
        batch_first: bool = True,
    ):

        super(LanguageEncoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embeded_size)
        if glove_word_emebd is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(glove_word_emebd))
        assert rnn_type in [
            "lstm",
            "gru",
        ], "rnn type {} can only be 'lstm' or 'gru'.".format(rnn_type)
        if rnn_type == "lstm":
            rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=batch_first,
            )
        elif rnn_type == "gru":
            rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                dropout=dropout,
                batch_first=batch_first,
            )
        self.encoder_rnn = DynamicRNN(rnn, output_last_layer=True)

    def foward(self, input_seq):
        seq_len = torch.sum(input_seq != 1, dim=1)
        word_emb = self.word_embedding(input_seq)
        output, (h, c) = self.encoder_rnn(word_emb, seq_len)
        return h, c


class Find(nn.Module):
    def __init__(
        self, query_size: int, key_size: int, mlp_mid_size: int, hidden_size: int
    ):
        super(Find, self).__init__()

        self.find_query_img_emb = FindModule(
            query_size=query_size,
            key_size=key_size,
            mlp_mid_size=mlp_mid_size,
            hidden_size=hidden_size,
        )

        self.find_query_kg_emb = FindModule(
            query_size=query_size,
            key_size=key_size,
            mlp_mid_size=mlp_mid_size,
            hidden_size=hidden_size,
        )

        self.match = Match(
            query_size=query_size,
            hidden_size=hidden_size,
            key_size=key_size,
            mid_size=mlp_mid_size,
        )

    def forward(self, query, img, kg, img_mask, kg_mask):

        _, lang_img_re = self.find_query_img_emb(query, img, img_mask)
        _, lang_kg_re = self.find_query_kg_emb(query, kg, kg_mask)
        img_map, kg_map, all_map = self.match(query, img, kg, lang_img_re, lang_kg_re)
        node_re = torch.matmul(all_map, torch.cat([img, kg]))
        return img_map, kg_map, all_map, node_re


class FindModule(nn.Module):
    def __init__(
        self, query_size: int, key_size: int, mlp_mid_size: int, hidden_size: int
    ):
        super(FindModule, self).__init__()

        self.query_mlp = MLP(
            in_size=query_size, mid_size=mlp_mid_size, out_size=hidden_size
        )

        self.key_mlp = MLP(
            in_size=key_size, mid_size=mlp_mid_size, out_size=hidden_size
        )

    def forward(self, query, key, mask=None):
        # query: batch_size, 1, query_node_size (query_node) ； key: batch_size, num_keys, key_size
        query = F.normalize(self.query_mlp(query), p=2, dim=1)
        key = F.normalize(self.key_mlp(key), p=2, dim=1)
        att_scores = torch.matmul(query, key.permute(0, 2, 1))

        if mask is not None:
            mask = mask.unsqueeze(1).expand(att_scores.shape)
            att_scores = att_scores.masked_fill(mask == 1, -1e32)

        att_map = F.softmax(att_scores, dim=1)
        value = torch.matmul(att_map, key)

        return att_map, value


class Match(nn.Module):
    def __init__(self, query_size: int, hidden_size: int, key_size: int, mid_size: int):
        super(Match, self).__init__()
        self.query_kg_gate = MLP(
            in_size=query_size + hidden_size, mid_size=mid_size, out_size=1
        )

        self.query_img_gate = MLP(in_size=query_size, mid_size=mid_size, out_size=1)

        self.hidden_mlp = MLP(
            in_size=hidden_size, mid_size=mid_size, out_size=hidden_size
        )

        self.key_mlp = MLP(in_size=key_size, mid_size=mid_size, out_size=hidden_size)

    def forward(self, query, img_re, kg_re, lang_img_re, lang_kg_re):
        # query: batch, 1, query_size
        # img_re : all nodes in image graph (batch, 36, feat_size)
        # kg_re: all nodes in kg graph (batch, num_kg, kg_feat_size)
        # lange_img_re: query_attend_img_out (batch, hidden_size)
        # lange_kg_re: query_attend_kg_out (batch, hidden_size)

        query_img_score = self.query_img_gate(torch.cat([query, lang_img_re]))
        query_kg_score = self.query_kg_gate(torch.cat([query, lang_kg_re]))
        score = F.softmax(torch.cat([query_img_score, query_kg_score], dim=1))
        node_re = torch.matmul(score, torch.cat([lang_img_re, lang_kg_re], dim=1))

        all_nodes = torch.cat([img_re, kg_re], dim=1)
        num_obj, _ = img_re.shape
        num_kg, _ = kg_re.shape
        hidden = F.normalize(self.hidden_mlp(node_re), p=2, dim=1)
        nodes = F.normalize(self.key_mlp(all_nodes, p=2, dim=1))
        all_node_map = F.softmax(torch.matmul(hidden, nodes.permute(0, 2, 1)))
        img_node_map = all_node_map[:num_obj]
        kg_node_map = all_node_map[num_obj:num_kg]

        return img_node_map, kg_node_map, all_node_map


class NodeTrans(nn.Module):
    def __init__(self, mid_size: int, hidden_size: int):
        super(NodeTrans, self).__init__()

        self.current_mlp = MLP(in_size=hidden_size, mid_size=mid_size, out_size=1)

    def forward(self, r_g, node_id, pre_node_id, img_graph, kg_graph):
        current_hidden = r_g.ndata["feat"][node_id]
        current_img_map = r_g.ndata["img_map"][node_id]
        current_kg_map = r_g.ndata["kg_map"][node_id]

        pre_img_map = r_g.ndata["img_map"][pre_node_id]
        pre_kg_map = r_g.ndata["kg_map"][pre_node_id]

        gate = F.sigmoid(self.current_mlp(current_hidden))

        new_img_map = current_img_map * gate + (1 - gate) * pre_img_map
        new_kg_map = current_kg_map * gate + (1 - gate) * pre_kg_map
        new_all_map = F.softmax(torch.cat[new_img_map, new_kg_map], dim=1)

        r_g.ndata["img_map"][node_id] = new_img_map
        r_g.ndata["kg_map"][node_id] = new_kg_map
        r_g.ndata["all_map"][node_id] = new_all_map

        all_node = torch.cat([img_graph.ndata["feat"], kg_graph.ndata["feat"]], dim=1)
        r_g.ndata["feat"][node_id] = torch.matmul(new_all_map, all_node)


class EdgeTrans(nn.Module):
    def __init__(self):
        super(EdgeTrans, self).__init__()

    def forward(
        self, r_g, img_edge_map, kg_edge_map, img_graph, kg_graph, node_id, pre_node_id
    ):

        self.update_img_node_map(r_g, img_edge_map, node_id, pre_node_id, img_graph)
        self.update_kg_node_map(r_g, kg_edge_map, node_id, pre_node_id, kg_graph)

        r_g.ndata["all_map"][node_id] = F.softmax(
            torch.cat(
                [r_g.ndata["img_map"][node_id], r_g.ndata["kg_map"][node_id]], dim=2
            )
        )

        r_g.ndata["feat"][node_id] = torch.matmul(
            r_g.ndata["all_map"][node_id],
            torch.cat([img_graph.ndata["feat"], kg_graph.ndata["feat"]]),
        )

    def update_img_node_map(self, r_g, edge_map, node_id, pre_node_id, img_graph):
        for n in range(img_graph.ndata["feat"].shape[0]):
            in_edges = img_graph.in_edge(n)
            if len(in_edges[0]) == 0:
                continue
            else:
                u, v = in_edges
                edges_ids = img_graph.edge_ids(u, v)
                r_g.ndata["img_map"][node_id][n] = torch.sum(
                    r_g.ndata["img_map"][pre_node_id][u] * edge_map[edges_ids], dim=1
                )

    def update_kg_node_map(self, r_g, edge_map, node_id, pre_node_id, kg_graph):
        # handle mask
        for n in range(kg_graph.ndata["feat"].shape[0]):
            in_edges = kg_graph.in_edge(n)
            if len(in_edges[0]) == 0:
                continue
            else:
                u, v = in_edges
                edges_ids = kg_graph.edge_ids(u, v)
                r_g.ndata["kg_map"][node_id][n] = torch.sum(
                    r_g.ndata["kg_map"][pre_node_id][u] * edge_map[edges_ids], dim=1
                )


class Reason(nn.Module):
    def __init__(self, hidden_size: int, mid_size: int, num_class: int):
        super(Reason, self).__init__()
        self.node_to_edge_linear = MLP(
            in_size=hidden_size * 2, mid_size=mid_size, out_size=hidden_size
        )
        self.ans_classifier = MLP(
            in_size=hidden_size, mid_size=mid_size, out_size=num_class
        )

    def forward(self, reason_graph, pre_node_ids):
        if len(pre_node_ids) > 1:
            pre_node_1, pre_node_2 = pre_node_ids[0]
            pre_node_1_id, pre_node_2_id = pre_node_1.item(), pre_node_2.item()
            # need to compute att feature then class
            out = self.node_to_edge_linear(
                torch.cat(
                    [
                        reason_graph.ndata["feat"][pre_node_1_id],
                        reason_graph.ndata["feat"][pre_node_2_id],
                    ],
                    dim=1,
                )
            )
        else:
            out = reason_graph.ndata["feat"]
        return self.ans_classifier(out)


# class ReasonNode(nn.Module):
#     # Given att_map over nodes , output prediction
#     def __init__(self,
#                  hidden_size: int,
#                  mid_size: int,
#                  num_class: int):
#         super(ReasonNode, self).__init__()
#         self.ans_classifier = MLP(in_size = hidden_size,
#                               mid_size = mid_size,
#                               out_size = num_class)
#
#     def forward(self,all_nodes, att_map):
#
#         out_emb = torch.matmul(att_map,all_nodes)
#         output = self.ans_classifier(out_emb)
#         return output
#
#
# class ReasonEdge(nn.Module):
#     # Gievn two node repre, output classifier
#
#     def __init__(self,
#                  hidden_size : int,
#                  mid_size: int,
#                  num_class: int):
#         super(ReasonEdge, self).__init__()
#
#         self.node_to_edge_linear = MLP(in_size = hidden_size*2,
#                                        mid_size = mid_size,
#                                        out_size = hidden_size)
#         self.ans_classifier = MLP(in_size = hidden_size,
#                                   mid_size = mid_size,
#                                   out_size = num_class)
#
#     def forward(self, node_1, node_2):
#
#         out_emb = self.node_to_edge_linear(torch.cat([node_1,node_2],dim=1))
#         out_put = self.ans_classifier(out_emb)
#
#         return  out_put
