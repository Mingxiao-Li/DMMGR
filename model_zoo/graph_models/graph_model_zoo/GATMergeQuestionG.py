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

@registry.register_model(name="GATMergeQuestionG")
class GATMergeQuestionG(nn.Module):
    def __init__(self, config):
        super(GATMergeQuestionG, self).__init__()
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


        self.question_node_proj = nn.Linear(
            config.LanguageEncoder.hidden_size, config.out_dim
        )
        self.question_edge_proj = nn.Linear(
            config.LanguageEncoder.hidden_size, config.embeded_size
        )

        self.edge_prj = nn.Linear(config.embeded_size, config.out_dim)
        self.mid_prj = nn.Linear(2 * config.out_dim, 3 * config.out_dim)
        self.pre_prj = nn.Linear(3 * config.out_dim, config.num_answer)

        # question guided img node attention
        self.img_node_att_proj_ques = nn.Linear(
            config.hidden_dim, config.img_node_att_ques_img_prj_dim
        )
        self.img_node_att_proj_img = nn.Linear(
            config.embeded_size, config.img_node_att_ques_img_prj_dim
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

        self.gat_1 = GAT(
            in_dim=config.in_dim,
            rel_dim=config.in_dim,
            hidden_dim=config.hidden_dim,
            out_dim=config.hidden_dim,
            num_heads=config.num_heads,
            layer_type=config.graph_layer_type,
        )

        self.gat_2 = GAT(
            in_dim = config.hidden_dim,
            rel_dim = config.hidden_dim,
            hidden_dim = config.hidden_dim,
            out_dim = config.hidden_dim,
            num_heads = config.num_heads,
            layer_type = config.graph_layer_type
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

        # question img node_att
        img_node_att_proj_ques_embed = self.img_node_att_proj_ques(question_embed)
        img_node_feat, img_edge_feat, num_img_nodes, num_img_edges = get_graph_feat(
            img_nodes, img_edges, self.word_embedding, self.max_img_node_len,self.max_img_edge_len,
            self.config.use_scene_graph
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

        # question kg node att
        kg_node_att_proj_ques_embed = self.kg_node_att_proj_ques(question_embed)
        kg_node_feat, kg_edge_feat, num_kg_nodes, num_kg_edges = get_graph_feat(
            kg_nodes, kg_edges, self.word_embedding, self.max_kg_len,self.max_kg_len,
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
            kg_edge_mask == 1, 1e-9
        )

        batch_size, _ = question.shape
        device = question.device

        graphs = []
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

            all_nodes_feat = torch.cat(
                [img_node_feat[b][:num_img_node], kg_node_feat[b][:num_kg_node]], dim=0
            )
            all_node_att_value = torch.cat(
                [img_node_att_value[b][:num_img_node], kg_node_att_value[b][:num_kg_node]], dim=0
            )
            all_node_att_value = F.softmax(all_node_att_value, dim=0)

            all_edges_feat = torch.cat(
                [img_edge_feat[b][:num_img_edge], kg_edge_feat[b][:num_kg_edge]], dim=0
            )
            all_edge_att_value = torch.cat(
                [img_edge_att_value[b][:num_img_edge], kg_edge_att_value[b][:num_kg_edge]], dim=0
            )
            all_edge_att_value = F.softmax(all_edge_att_value, dim=0)

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
                num_nodes = num_img_node + num_kg_node,
            )
            g = g.to(device)
            g.ndata["feat"] = all_nodes_feat
            g.ndata["att"] = all_node_att_value
            g.edata["feat"] = all_edges_feat
            g.edata["att"] = all_edge_att_value
            graphs.append(g)
        
        graph_batch = dgl.batch(graphs)
        graph_batch = self.gat_1(graph_batch)
 
        graph_batch = self.gat_2(graph_batch)

        graphs  = dgl.unbatch(graph_batch)
        nodes = []
        edges = []

        for i in range(batch_size):
            nodes.append(torch.sum(graphs[i].ndata["feat"], dim=0).unsqueeze(0))
            edges.append(torch.sum(graphs[i].edata["feat"], dim=0).unsqueeze(0))
        
        nodes = torch.cat(nodes,dim=0)
        edges = torch.cat(edges,dim=0)

        question_edge = self.question_edge_proj(question_embed)
        question_node = self.question_node_proj(question_embed)

        q_edge = question_edge + edges
        q_node = question_node + nodes
        
        q_edge = self.edge_prj(q_edge)
        q_pre = torch.cat([q_edge, q_node], dim=1)
        q_mid = self.mid_prj(q_pre)
        pred = self.pre_prj(q_mid)

        return pred 

