import torch
import os
import dgl
import json
import torch.nn as nn
import torch.nn.functional as F
from model_zoo.graph_models.graph_model_zoo.graph_modules import (
    RGraphAttention,
    RCrossReason,
    AttentionAggretate,
    PredictionHead,
    LanguageEncoder,
)
from x.core.registry import registry
from x.common.util import get_numpy_word_embed
import pdb


@registry.register_model(name="CGRM")
class CGRM(nn.Module):
    def __init__(self, config):
        super(CGRM, self).__init__()
        self.config = config
        self.max_kg_nodes = config.max_kg_nodes
        self.max_kg_edges = config.max_kg_edges
        self.max_img_nodes = config.max_img_nodes
        self.max_img_edges = config.max_img_edges
        self.max_kg_len = config.max_kg_ele_len

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

        self.language_encoder = LanguageEncoder(config.LanguageEncoder)
        self.ques_img_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.ques_kg_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.ques_img_att_proj = nn.Linear(config.hidden_size * 2, 1)
        self.ques_kg_att_proj = nn.Linear(config.hidden_size * 2, 1)

        self.img_node_proj = nn.Linear(config.img_size, config.hidden_size)
        self.img_edge_proj = nn.Linear(config.img_loc_size, config.hidden_size)

        self.kg_node_proj = nn.Linear(config.kg_size, config.hidden_size)
        self.kg_edge_proj = nn.Linear(config.kg_size, config.hidden_size)

        self.img_graph_reason = RGraphAttention(
            hidden_size=config.hidden_size, att_fuse_method=config.att_fuse_method
        )

        self.img_graph_aggregate = AttentionAggretate(
            hidden_size=config.hidden_size,
            max_nodes=self.max_img_nodes,
            max_edges=self.max_img_edges,
        )

        self.kg_graph_reason = RGraphAttention(
            hidden_size=config.hidden_size, att_fuse_method=config.att_fuse_method
        )

        self.kg_graph_aggregate = AttentionAggretate(
            hidden_size=config.hidden_size,
            max_nodes=self.max_kg_nodes,
            max_edges=self.max_kg_edges,
        )

        self.img_to_kg_cross_reason = RCrossReason(
            hidden_size=config.hidden_size, att_fuse_method=config.att_fuse_method
        )
        self.kg_graph_rel_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.kg_graph_cross_aggregate = AttentionAggretate(
            hidden_size=config.hidden_size,
            max_nodes=self.max_kg_nodes,
            max_edges=self.max_kg_edges,
        )

        self.kg_to_img_cross_reason = RCrossReason(
            hidden_size=config.hidden_size, att_fuse_method=config.att_fuse_method
        )
        self.img_graph_rel_proj = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.img_graph_cross_aggregate = AttentionAggretate(
            hidden_size=config.hidden_size,
            max_nodes=self.max_img_nodes,
            max_edges=self.max_img_edges,
        )

        self.img_gate_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.kg_gate_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.predict_gate = nn.Linear(
            config.hidden_size, config.hidden_size, bias=False
        )
        self.sigmoid = nn.Sigmoid()

        self.prediction_head = nn.Linear(
            config.hidden_size, config.num_answer, bias=False
        )

    def forward(
        self,
        question,
        question_mask,
        img_feat,
        img_loc,
        img_node1_id_list,
        img_node2_id_list,
        kg_entity,
        kg_edge,
        kg_node1_ids_list,
        kg_node2_ids_list,
    ):
        r"""
        img_feat: shape=batch, num_obj, 2048
        img_loc: relative position shape=batch, num_edge, 5
        img_node1_id_list: shape=batch num_edge
        img_node2_id_list: shape=batch num_edge
        """
        device = question.device
        seq_word_emb = self.word_embedding(question)
        seq_len = torch.sum(question_mask == 1, dim=1)
        all_out, lens, (hidden_out, cell_out) = self.language_encoder(
            seq_word_emb, seq_len
        )
        ques_img = self.ques_img_proj(hidden_out)
        ques_kg = self.ques_kg_proj(hidden_out)
        # attention over words
        ques_img_score = self.ques_img_att_proj(
            torch.cat([all_out, ques_img.unsqueeze(1).expand_as(all_out)], dim=2)
        )
        ques_img_score = ques_img_score.squeeze(2).masked_fill(
            question_mask == 1, -1e32
        )
        ques_img_score = F.softmax(ques_img_score, dim=1)
        ques_kg_score = self.ques_kg_att_proj(
            torch.cat([all_out, ques_kg.unsqueeze(1).expand_as(all_out)], dim=2)
        )
        ques_kg_score = ques_kg_score.squeeze(2).masked_fill(question_mask == 1, -1e32)
        ques_kg_score = F.softmax(ques_kg_score, dim=1)

        ques_img = torch.matmul(ques_img_score.unsqueeze(1), all_out).squeeze(1)
        ques_kg = torch.matmul(ques_kg_score.unsqueeze(1), all_out).squeeze(1)

        # build img graph
        batch_size, num_obj, _ = img_feat.shape
        img_node_feat = self.img_node_proj(img_feat)
        img_edge_feat = self.img_edge_proj(img_loc)

        img_graphs = []

        for b in range(batch_size):
            g = dgl.DGLGraph()
            g = g.to(device)
            g.add_nodes(num_obj)
            g.add_edge(img_node1_id_list[b], img_node2_id_list[b])
            g.ndata["feat"] = img_node_feat[b]
            g.edata["feat"] = img_edge_feat[b]
            g.edata["guild_vec"] = ques_img[b].expand_as(img_edge_feat[b])
            img_graphs.append(g)
        img_graph_batch = dgl.batch(img_graphs)
        # build kg graph
        # _, num_kg_entity, _ = kg_entity.shape   # batch, max_num_entity (pad), entity_len(pad)
        num_entites = torch.sum(torch.sum(kg_entity, dim=2) != self.max_kg_len, dim=1)
        num_edges = torch.sum(torch.sum(kg_edge, dim=2) != self.max_kg_len, dim=1)

        entity_rel_len = torch.sum(kg_entity != 1, dim=2)
        entity_rel_len += entity_rel_len == 0
        edge_rel_len = torch.sum(kg_edge != 1, dim=2)
        edge_rel_len += edge_rel_len == 0

        entity_embedding = self.word_embedding(kg_entity)
        edge_embedding = self.word_embedding(kg_edge)
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
        entity_feat = self.kg_node_proj(entity_feat)
        edge_feat = self.kg_edge_proj(edge_feat)

        kg_graphs = []
        for b in range(batch_size):
            num_entity = num_entites[b].item()
            num_edge = num_edges[b].item()
            g = dgl.DGLGraph()
            g = g.to(device)
            g.add_nodes(num_entity)
            g.add_edge(kg_node1_ids_list[b], kg_node2_ids_list[b])
            g.ndata["feat"] = entity_feat[b][:num_entity]
            g.edata["feat"] = edge_feat[b][:num_edge]
            g.edata["guild_vec"] = ques_kg[b].expand_as(
                edge_feat[b][:num_edge]
            )  # ques guild or img_node_out_1 guild
            kg_graphs.append(g)
        kg_graph_batch = dgl.batch(kg_graphs)

        # reason over img_graph
        img_graph_batch = self.img_graph_reason(img_graph_batch)
        # aggregate
        img_graph_out_1 = self.img_graph_aggregate(img_graph_batch, ques_img)

        # reason over kg_graph
        kg_graph_batch = self.kg_graph_reason(kg_graph_batch)

        # aggregate
        kg_graph_out_1 = self.kg_graph_aggregate(kg_graph_batch, ques_kg)

        # proj aggregate
        img_rel_vec = self.img_graph_rel_proj(
            torch.cat([ques_img, kg_graph_out_1], dim=1)
        )
        kg_rel_vec = self.kg_graph_rel_proj(
            torch.cat([ques_kg, img_graph_out_1], dim=1)
        )

        # update guild feat in kg_graph and img_graph
        img_graphs_unbatch = dgl.unbatch(img_graph_batch)
        kg_graphs_unbatch = dgl.unbatch(kg_graph_batch)
        for b in range(batch_size):
            num_img_edges, dim = img_graphs_unbatch[b].edata["guild_vec"].shape
            img_graphs_unbatch[b].edata["guild_vec"] = (
                img_rel_vec[b].unsqueeze(0).expand(num_img_edges, dim)
            )
            num_kg_edges, dim = kg_graphs_unbatch[b].edata["guild_vec"].shape
            kg_graphs_unbatch[b].edata["guild_vec"] = (
                kg_rel_vec[b].unsqueeze(0).expand(num_kg_edges, dim)
            )
        img_graph_batch = dgl.batch(img_graphs_unbatch)
        kg_graph_batch = dgl.batch(kg_graphs_unbatch)

        img_graph_batch = self.kg_to_img_cross_reason(img_graph_batch)
        # aggregate
        img_graph_out_2 = self.img_graph_cross_aggregate(img_graph_batch, img_rel_vec)

        # cross reason img to kg reason
        kg_graph_batch = self.img_to_kg_cross_reason(kg_graph_batch)
        # aggregate
        kg_graph_out_2 = self.kg_graph_cross_aggregate(kg_graph_batch, kg_rel_vec)

        # skip connect
        img_gate = self.sigmoid(F.leaky_relu(self.img_gate_proj(img_graph_out_2)))
        kg_gate = self.sigmoid(F.leaky_relu(self.kg_gate_proj(kg_graph_out_2)))

        img_out = img_gate * img_graph_out_1 + (1 - img_gate) * img_graph_out_2
        kg_out = kg_gate * kg_graph_out_1 + (1 - kg_gate) * kg_graph_out_2

        pre_gate = self.sigmoid(F.leaky_relu(self.predict_gate(img_out)))
        out = pre_gate * img_out + (1 - pre_gate) * kg_out
        prediction = self.prediction_head(out)
        return prediction