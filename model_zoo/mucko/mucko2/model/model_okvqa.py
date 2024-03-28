import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import numpy as numpy
from util.dynamic_rnn import DynamicRNN
from model.img_gcn import ImageGCN
from model.semantic_gcn import SemanticGCN
from model.cross_gcn import CrossGCN
from model.fact_gcn import FactGCN

import dgl
import networkx as nx
import numpy as np


class CMGCNnet(nn.Module):
    def __init__(self, config, que_vocabulary, glove, device):
        '''
        :param config: 配置参数
        :param que_vocabulary: 字典 word 2 index
        :param glove: (voc_size,embed_size)
        '''
        super(CMGCNnet, self).__init__()
        self.config = config
        self.device = device
        
        self.que_glove_embed = nn.Embedding(len(que_vocabulary), config['model']['glove_embedding_size'])
       
        self.que_glove_embed.weight.data = glove

        self.que_glove_embed.weight.requires_grad = False


        self.ques_rnn = nn.LSTM(config['model']['glove_embedding_size'],
                                config['model']['lstm_hidden_size'],
                                config['model']['lstm_num_layers'],
                                batch_first=True,
                                dropout=config['model']['dropout'])
        self.ques_rnn = DynamicRNN(self.ques_rnn)


        self.vis_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.vis_node_att_proj_img = nn.Linear(
            config['model']['img_feature_size'],
            config['model']['node_att_ques_img_proj_dims'])
        self.vis_node_att_value = nn.Linear(
            config['model']['node_att_ques_img_proj_dims'], 1)


        self.vis_node_att_proj_ques2 = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['node_att_ques_img_proj_dims2'])
        self.vis_node_att_proj_img2 = nn.Linear(
            config['model']['image_gcn1_out_dim'],
            config['model']['node_att_ques_img_proj_dims2'])
        self.vis_node_att_value2 = nn.Linear(
            config['model']['node_att_ques_img_proj_dims2'], 1)


        self.vis_rel_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.vis_rel_att_proj_rel = nn.Linear(
            config['model']['vis_relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.vis_rel_att_value = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims'], 1)


        self.vis_rel_att_proj_ques2 = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims2'])
        self.vis_rel_att_proj_rel2 = nn.Linear(
            config['model']['vis_relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims2'])
        self.vis_rel_att_value2 = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims2'], 1)


        self.sem_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['sem_node_att_ques_img_proj_dims'])
        self.sem_node_att_proj_sem = nn.Linear(
            config['model']['sem_node_dims'],
            config['model']['sem_node_att_ques_img_proj_dims'])
        self.sem_node_att_value = nn.Linear(
            config['model']['sem_node_att_ques_img_proj_dims'], 1)


        self.sem_rel_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.sem_rel_att_proj_rel = nn.Linear(
            config['model']['sem_relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims'])
        self.sem_rel_att_value = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims'], 1)


        self.sem_node_att_proj_ques2 = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['sem_node_att_ques_img_proj_dims2'])
        self.sem_node_att_proj_sem2 = nn.Linear(
            config['model']['semantic_gcn1_out_dim'],
            config['model']['sem_node_att_ques_img_proj_dims2'])
        self.sem_node_att_value2 = nn.Linear(
            config['model']['sem_node_att_ques_img_proj_dims2'], 1)


        self.sem_rel_att_proj_ques2 = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['rel_att_ques_rel_proj_dims2'])
        self.sem_rel_att_proj_rel2 = nn.Linear(
            config['model']['sem_relation_dims'],
            config['model']['rel_att_ques_rel_proj_dims2'])
        self.sem_rel_att_value2 = nn.Linear(
            config['model']['rel_att_ques_rel_proj_dims2'], 1)


        self.fact_node_att_proj_ques = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['fact_node_att_ques_node_proj_dims'])
        self.fact_node_att_proj_node = nn.Linear(
            config['model']['fact_node_dims'],
            config['model']['fact_node_att_ques_node_proj_dims'])
        self.fact_node_att_value = nn.Linear(
            config['model']['fact_node_att_ques_node_proj_dims'], 1)


        self.fact_node_att_proj_ques2 = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['fact_node_att_ques_node_proj_dims2'])
        self.fact_node_att_proj_node2 = nn.Linear(
            config['model']['cross_gcn1_out_dim'],
            config['model']['fact_node_att_ques_node_proj_dims2'])
        self.fact_node_att_value2 = nn.Linear(
            config['model']['fact_node_att_ques_node_proj_dims2'], 1)


        self.fact_node_att_proj_ques3 = nn.Linear(
            config['model']['lstm_hidden_size'],
            config['model']['fact_node_att_ques_node_proj_dims3'])
        self.fact_node_att_proj_node3 = nn.Linear(
            config['model']['cross_gcn2_out_dim'],
            config['model']['fact_node_att_ques_node_proj_dims3'])
        self.fact_node_att_value3 = nn.Linear(
            config['model']['fact_node_att_ques_node_proj_dims3'], 1)


        self.img_gcn1 = ImageGCN(config,
                                 in_dim=config['model']['img_feature_size'],
                                 out_dim=config['model']['image_gcn1_out_dim'],
                                 rel_dim=config['model']['vis_relation_dims'])


        self.sem_gcn1 = SemanticGCN(config,
                                    in_dim=config['model']['sem_node_dims'],
                                    out_dim=config['model']['semantic_gcn1_out_dim'],
                                    rel_dim=config['model']['sem_relation_dims'])


        self.fact_gcn1 = FactGCN(config,
                                 in_dim=config['model']['fact_node_dims'],
                                 out_dim=config['model']['fact_gcn1_out_dim'])


        self.img_gcn2 = ImageGCN(config,
                                 in_dim=config['model']['image_gcn1_out_dim'],
                                 out_dim=config['model']['image_gcn2_out_dim'],
                                 rel_dim=config['model']['vis_relation_dims'])


        self.sem_gcn2 = SemanticGCN(config,
                                    in_dim=config['model']['semantic_gcn1_out_dim'],
                                    out_dim=config['model']['semantic_gcn2_out_dim'],
                                    rel_dim=config['model']['sem_relation_dims'])


        self.fact_gcn2 = FactGCN(config,
                                 in_dim=config['model']['fact_gcn1_out_dim'],
                                 out_dim=config['model']['fact_gcn2_out_dim'])

        self.cross_gcn = CrossGCN(
            config,
            in_dims=config['model']['fact_gcn1_out_dim'],
            out_dims=config['model']['cross_gcn1_out_dim'],
            img_att_proj_dim=config['model']['cross_gcn1_img_att_proj_dim'],
            sem_att_proj_dim=config['model']['cross_gcn1_sem_att_proj_dim'],
            img_dim=config['model']['image_gcn1_out_dim'],
            sem_dim=config['model']['semantic_gcn1_out_dim'],
            # fact_dim=config['model']['cross_gcn1_fact_out_dim'],
            gate_dim=config['model']['cross_gcn1_gate_dim'],
            que_dims=config['model']['lstm_hidden_size'])

        self.cross_gcn2 = CrossGCN(
            config,
            in_dims=config['model']['fact_gcn2_out_dim'],
            out_dims=config['model']['cross_gcn2_out_dim'],
            img_att_proj_dim=config['model']['cross_gcn2_img_att_proj_dim'],
            sem_att_proj_dim=config['model']['cross_gcn2_sem_att_proj_dim'],
            img_dim=config['model']['image_gcn2_out_dim'],
            sem_dim=config['model']['semantic_gcn2_out_dim'],
            # fact_dim=config['model']['cross_gcn2_fact_out_dim'],
            gate_dim=config['model']['cross_gcn2_gate_dim'],
            que_dims=config['model']['lstm_hidden_size'])


        self.fact_gcn3 = FactGCN(config,
                                 in_dim=config['model']['cross_gcn2_out_dim'],
                                 out_dim=config['model']['fact_gcn3_out_dim'])

        self.mlp = nn.Sequential(
            nn.Linear(config['model']['cross_gcn2_out_dim'] + config['model']['lstm_hidden_size'], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, batch):

        batch_size = len(batch['id_list'])


        images = batch['features_list']  
        images = torch.stack(images).to(self.device) 

        img_relations = batch['img_relations_list']
        img_relations = torch.stack(img_relations).to(self.device)  


        questions = batch['question_list']  
        questions = torch.stack(questions).to(self.device)  

        questions_len_list = batch['question_length_list']
        questions_len_list = torch.Tensor(questions_len_list).long().to(self.device)


        semantic_num_nodes_list = torch.Tensor(batch['semantic_num_nodes_list']).long().to(self.device)  # (bs,

        semantic_n_features_list = batch['semantic_node_features_list']
        semantic_n_features_list = [features.to(self.device) for features in semantic_n_features_list]  # [(num_nodes, 300)]

        semantic_e1ids_list = batch['semantic_e1ids_list']
        semantic_e1ids_list = [e1ids.to(self.device) for e1ids in semantic_e1ids_list]  # [(num_edges)]

        semantic_e2ids_list = batch['semantic_e2ids_list']
        semantic_e2ids_list = [e2ids.to(self.device) for e2ids in semantic_e2ids_list]  # [(num_edges)]

        semantic_e_features_list = batch['semantic_edge_features_list']
        semantic_e_features_list = [features.to(self.device) for features in semantic_e_features_list]  # [(num_edges,300)]

        # fact graph
        fact_num_nodes_list = torch.Tensor(batch['facts_num_nodes_list']).long().to(self.device)

        facts_features_list = batch['facts_node_features_list']
        facts_features_list = [features.to(self.device) for features in facts_features_list]

        facts_e1ids_list = batch['facts_e1ids_list']
        facts_e1ids_list = [e1ids.to(self.device) for e1ids in facts_e1ids_list]

        facts_e2ids_list = batch['facts_e2ids_list']
        facts_e2ids_list = [e2ids.to(self.device) for e2ids in facts_e2ids_list]

        facts_answer_list = batch['facts_answer_list']
        facts_answer_list = [answer.to(self.device) for answer in facts_answer_list]

        facts_answer_id_list = torch.Tensor(batch['facts_answer_id_list']).long().to(self.device)

        ques_embed = self.que_glove_embed(questions).float()  # shape (bs,max_length,300)
        _, (ques_embed, _) = self.ques_rnn(ques_embed, questions_len_list)  # qes_embed shape=(batch,hidden_size)

        node_att_proj_ques_embed = self.vis_node_att_proj_ques(ques_embed)  # shape (batch,proj_size)
        node_att_proj_img_embed = self.vis_node_att_proj_img(images)  # shape (batch,36,proj_size)
        
        node_att_proj_ques_embed = node_att_proj_ques_embed.unsqueeze(1).repeat(1, images.shape[1],
                                                                                1)  # shape(batch,36,proj_size)
        node_att_proj_img_sum_ques = torch.tanh(node_att_proj_ques_embed + node_att_proj_img_embed)
        vis_node_att_values = self.vis_node_att_value(node_att_proj_img_sum_ques).squeeze()  # shape(batch,36)
        vis_node_att_values = F.softmax(vis_node_att_values, dim=-1)  # shape(batch,36)

        rel_att_proj_ques_embed = self.vis_rel_att_proj_ques(ques_embed)  # shape(batch,128)
        rel_att_proj_rel_embed = self.vis_rel_att_proj_rel(img_relations)  # shape(batch,36,36,128)

        rel_att_proj_ques_embed = rel_att_proj_ques_embed.repeat(
            1, 36 * 36).view(
            batch_size, 36, 36, self.config['model']
            ['rel_att_ques_rel_proj_dims'])  # shape(batch,36,36,128)
        rel_att_proj_rel_sum_ques = torch.tanh(rel_att_proj_ques_embed +
                                               rel_att_proj_rel_embed)
        vis_rel_att_values = self.vis_rel_att_value(rel_att_proj_rel_sum_ques).squeeze()  # shape(batch,36,36)

        sem_node_att_val_list = []
        sem_edge_att_val_list = []
        for i in range(batch_size):
            num_node = semantic_num_nodes_list[i]  # n
            sem_node_features = semantic_n_features_list[i]  
            q_embed = ques_embed[i]  
            q_embed = q_embed.repeat(num_node, 1)  
            sem_node_att_proj_ques_embed = self.sem_node_att_proj_ques(q_embed) 
            sem_node_att_proj_sem_embed = self.sem_node_att_proj_sem(sem_node_features) 
            sem_node_att_proj_sem_sum_ques = torch.tanh(
                sem_node_att_proj_ques_embed + sem_node_att_proj_sem_embed) 
            sem_node_att_values = self.sem_node_att_value(sem_node_att_proj_sem_sum_ques)  
            sem_node_att_values = F.softmax(sem_node_att_values, dim=0)  

            sem_node_att_val_list.append(sem_node_att_values)

            num_edge = semantic_e_features_list[i].shape[0] 
            sem_edge_features = semantic_e_features_list[i]  
            qq_embed = ques_embed[i]  # (512)
            qq_embed = qq_embed.unsqueeze(0).repeat(num_edge, 1)  
            sem_rel_att_proj_ques_embed = self.sem_rel_att_proj_ques(qq_embed)  
            sem_rel_att_proj_rel_embed = self.sem_rel_att_proj_rel(sem_edge_features)  
            sem_rel_att_proj_rel_sum_ques = torch.tanh(
                sem_rel_att_proj_ques_embed + sem_rel_att_proj_rel_embed)  
            sem_rel_att_values = self.sem_rel_att_value(sem_rel_att_proj_rel_sum_ques)  
            sem_rel_att_values = F.softmax(sem_rel_att_values, dim=0)  

            sem_edge_att_val_list.append(sem_rel_att_values)

        fact_node_att_values_list = []
        for i in range(batch_size):
            num_node = facts_features_list[i].shape[0]  # n
            fact_node_features = facts_features_list[i] 
            q_embed = ques_embed[i]  
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  
            fact_node_att_proj_ques_embed = self.fact_node_att_proj_ques(q_embed)  
            fact_node_att_proj_node_embed = self.fact_node_att_proj_node(fact_node_features)  
            fact_node_att_proj_node_sum_ques = torch.tanh(
                fact_node_att_proj_ques_embed + fact_node_att_proj_node_embed)  
            fact_node_att_values = self.fact_node_att_value(fact_node_att_proj_node_sum_ques) 
            fact_node_att_values = F.softmax(fact_node_att_values, dim=0) 
            fact_node_att_values_list.append(fact_node_att_values)

        
        img_graphs = []
        for i in range(batch_size):
            g = dgl.DGLGraph()

            g.add_nodes(36)

            g.ndata['h'] = images[i]
            g.ndata['att'] = vis_node_att_values[i].unsqueeze(-1)
            g.ndata['batch'] = torch.full([36, 1], i)

            for s in range(36):
                for d in range(36):
                    g.add_edge(s, d)

            g.edata['rel'] = img_relations[i].view(36 * 36, self.config['model']['vis_relation_dims'])  # shape(36*36,7)
            g.edata['att'] = vis_rel_att_values[i].view(36 * 36, 1)  # shape(36*36,1)
            img_graphs.append(g)
        image_batch_graph = dgl.batch(img_graphs)

        semantic_graphs = []
        for i in range(batch_size):
            graph = dgl.DGLGraph()
            graph.add_nodes(semantic_num_nodes_list[i])
            graph.add_edges(semantic_e1ids_list[i], semantic_e2ids_list[i])
            graph.ndata['h'] = semantic_n_features_list[i]
            graph.ndata['att'] = sem_node_att_val_list[i]
            graph.edata['r'] = semantic_e_features_list[i]
            graph.edata['att'] = sem_edge_att_val_list[i]
            semantic_graphs.append(graph)
        semantic_batch_graph = dgl.batch(semantic_graphs)

        fact_graphs = []
        for i in range(batch_size):
            graph = dgl.DGLGraph()
            graph.add_nodes(facts_features_list[i].shape[0])
            graph.add_edges(facts_e1ids_list[i], facts_e2ids_list[i])
            graph.ndata['h'] = facts_features_list[i]
            graph.ndata['att'] = fact_node_att_values_list[i]
            graph.ndata['batch'] = torch.full([fact_num_nodes_list[i], 1], i)
            graph.ndata['answer'] = facts_answer_list[i]
            fact_graphs.append(graph)
        fact_batch_graph = dgl.batch(fact_graphs)

        
        image_batch_graph = self.img_gcn1(image_batch_graph)

        semantic_batch_graph = self.sem_gcn1(semantic_batch_graph)

        fact_batch_graph = self.fact_gcn1(fact_batch_graph)

        fact_batch_graph = self.cross_gcn(fact_batch_graph, image_batch_graph, semantic_batch_graph, ques_embed)

        images = []
        img_graphs = dgl.unbatch(image_batch_graph)
        for img_graph in img_graphs:
            images.append(img_graph.ndata['h'])
        images = torch.stack(images)  # (bs,36,1024)

        node_att_proj_ques_embed = self.vis_node_att_proj_ques2(ques_embed)  # shape (batch,proj_size)
        node_att_proj_img_embed = self.vis_node_att_proj_img2(images)  # shape (batch,36,proj_size)
        
        node_att_proj_ques_embed = node_att_proj_ques_embed.unsqueeze(1).repeat(1, images.shape[1], 1)  # shape(batch,36,proj_size)
        node_att_proj_img_sum_ques = torch.tanh(node_att_proj_ques_embed + node_att_proj_img_embed)
        vis_node_att_values = self.vis_node_att_value2(node_att_proj_img_sum_ques).squeeze()  # shape(batch,36)
        vis_node_att_values = F.softmax(vis_node_att_values, dim=-1)  # shape(batch,36)

        rel_att_proj_ques_embed = self.vis_rel_att_proj_ques2(ques_embed)  # shape(batch,128)
        rel_att_proj_rel_embed = self.vis_rel_att_proj_rel2(img_relations)  # shape(batch,36,36,128)

        rel_att_proj_ques_embed = rel_att_proj_ques_embed.repeat(
            1, 36 * 36).view(
            batch_size, 36, 36, self.config['model']
            ['rel_att_ques_rel_proj_dims2'])  # shape(batch,36,36,128)
        rel_att_proj_rel_sum_ques = torch.tanh(rel_att_proj_ques_embed +
                                               rel_att_proj_rel_embed)
        vis_rel_att_values = self.vis_rel_att_value2(rel_att_proj_rel_sum_ques).squeeze()  # shape(batch,36,36)

        for i, img_graph in enumerate(img_graphs):
            img_graph.ndata['att'] = vis_node_att_values[i].unsqueeze(-1)
            img_graph.edata['att'] = vis_rel_att_values[i].view(36 * 36, 1)

        
        sem_node_att_val_list = []
        sem_edge_att_val_list = []
        semantic_graphs = dgl.unbatch(semantic_batch_graph)
        for i in range(batch_size):
            num_node = semantic_graphs[i].number_of_nodes()  # n
            sem_node_features = semantic_graphs[i].ndata['h']  # (n,300)
            q_embed = ques_embed[i]  # (512)
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  # (n,512)
            sem_node_att_proj_ques_embed = self.sem_node_att_proj_ques2(q_embed)  # shape (n,p)
            sem_node_att_proj_sem_embed = self.sem_node_att_proj_sem2(sem_node_features)  # shape (n,p)
            sem_node_att_proj_sem_sum_ques = torch.tanh(
                sem_node_att_proj_ques_embed + sem_node_att_proj_sem_embed)  # shape (n,p)
            sem_node_att_values = self.sem_node_att_value2(sem_node_att_proj_sem_sum_ques)  # shape(n,1)
            sem_node_att_values = F.softmax(sem_node_att_values, dim=0)  # shape(n,1)
            semantic_graphs[i].ndata['att'] = sem_node_att_values

            sem_node_att_val_list.append(sem_node_att_values)

            num_edge = semantic_e_features_list[i].shape[0]  # n
            sem_edge_features = semantic_e_features_list[i]  # (n,300)
            qq_embed = ques_embed[i]  # (512)
            qq_embed = qq_embed.unsqueeze(0).repeat(num_edge, 1)  # (n,512)
            sem_rel_att_proj_ques_embed = self.sem_rel_att_proj_ques2(qq_embed)  # shape (n,p)
            sem_rel_att_proj_rel_embed = self.sem_rel_att_proj_rel2(sem_edge_features)  # shape (n,p)
            sem_rel_att_proj_rel_sum_ques = torch.tanh(
                sem_rel_att_proj_ques_embed + sem_rel_att_proj_rel_embed)  # shape (n,p)
            sem_rel_att_values = self.sem_rel_att_value2(sem_rel_att_proj_rel_sum_ques)  # shape(n,1)
            sem_rel_att_values = F.softmax(sem_rel_att_values, dim=0)  # shape(n,1)
            semantic_graphs[i].edata['att'] = sem_rel_att_values

            sem_edge_att_val_list.append(sem_rel_att_values)

        fact_node_att_values_list = []
        fact_graphs = dgl.unbatch(fact_batch_graph)
        for i in range(batch_size):
            num_node = facts_features_list[i].shape[0]  # n
            fact_node_features = fact_graphs[i].ndata['h']  # (n,300)
            q_embed = ques_embed[i]  # (512)
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  # (n,512)
            fact_node_att_proj_ques_embed = self.fact_node_att_proj_ques2(q_embed)  # shape (n,p)
            fact_node_att_proj_node_embed = self.fact_node_att_proj_node2(fact_node_features)  # shape (n,p)
            fact_node_att_proj_node_sum_ques = torch.tanh(
                fact_node_att_proj_ques_embed + fact_node_att_proj_node_embed)  # shape (n,p)
            fact_node_att_values = self.fact_node_att_value2(fact_node_att_proj_node_sum_ques)  # shape(n,1)
            fact_node_att_values = F.softmax(fact_node_att_values, dim=0)  # shape(n,1)
            fact_graphs[i].ndata['att'] = fact_node_att_values

            fact_node_att_values_list.append(fact_node_att_values)

        
        image_batch_graph = self.img_gcn2(image_batch_graph)
        
        semantic_batch_graph = self.sem_gcn2(semantic_batch_graph)
        
        fact_batch_graph = self.fact_gcn2(fact_batch_graph)

        fact_batch_graph = self.cross_gcn2(fact_batch_graph, image_batch_graph, semantic_batch_graph, ques_embed)

        fact_node_att_values_list = []
        fact_graphs = dgl.unbatch(fact_batch_graph)
        for i in range(batch_size):
            num_node = facts_features_list[i].shape[0]  # n
            fact_node_features = fact_graphs[i].ndata['h']  # (n,300)
            q_embed = ques_embed[i]  # (512)
            q_embed = q_embed.unsqueeze(0).repeat(num_node, 1)  # (n,512)
            fact_node_att_proj_ques_embed = self.fact_node_att_proj_ques3(q_embed)  # shape (n,p)
            fact_node_att_proj_node_embed = self.fact_node_att_proj_node3(fact_node_features)  # shape (n,p)
            fact_node_att_proj_node_sum_ques = torch.tanh(
                fact_node_att_proj_ques_embed + fact_node_att_proj_node_embed)  # shape (n,p)
            fact_node_att_values = self.fact_node_att_value3(fact_node_att_proj_node_sum_ques)  # shape(n,1)
            fact_node_att_values = F.softmax(fact_node_att_values, dim=0)  # shape(n,1)
            fact_graphs[i].ndata['att'] = fact_node_att_values
            fact_node_att_values_list.append(fact_node_att_values)

        fact_graphs = dgl.unbatch(fact_batch_graph)
        new_fact_graphs = []
        for i, fact_graph in enumerate(fact_graphs):
            num_nodes = fact_graph.number_of_nodes()
            q_embed = ques_embed[i]
            q_embed = q_embed.unsqueeze(0).repeat(num_nodes, 1)
            fact_graph.ndata['h'] = torch.cat([fact_graph.ndata['h'], q_embed], dim=1)
            new_fact_graphs.append(fact_graph)
        fact_batch_graph = dgl.batch(new_fact_graphs)

        fact_batch_graph.ndata['h'] = self.mlp(fact_batch_graph.ndata['h'])

        return fact_batch_graph
