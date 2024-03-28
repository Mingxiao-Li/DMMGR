import dgl
import torch

import torch.nn.functional as F
from torch import nn
import numpy as np
import dgl.function as fn


class CrossGCN(nn.Module):
    def __init__(self, config, in_dims, out_dims, img_att_proj_dim, sem_att_proj_dim, img_dim, sem_dim,
                 gate_dim, que_dims):
        super(CrossGCN, self).__init__()
        self.gcn = CrossGCNLayer(config, in_dims, out_dims, img_att_proj_dim, sem_att_proj_dim, img_dim, sem_dim, gate_dim, que_dims)
    
    def forward(self, fact_batch_graph, img_batch_graph, sem_batch_graph, ques_embed):
        fact_graphs = dgl.unbatch(fact_batch_graph)
        img_graphs = dgl.unbatch(img_batch_graph)
        sem_graphs = dgl.unbatch(sem_batch_graph)
        num_graph = len(fact_graphs)
        new_fact_graphs = []
        for i in range(num_graph):
            fact_graph = fact_graphs[i]
            img_graph = img_graphs[i]
            sem_graph = sem_graphs[i]
            que_embed = ques_embed[i]
            fact_graph = self.gcn(fact_graph, img_graph, sem_graph, que_embed)

            new_fact_graphs.append(fact_graph)
        return dgl.batch(new_fact_graphs)


class CrossGCNLayer(nn.Module):
    def __init__(self, config, in_dims, out_dims, img_att_proj_dim, sem_att_proj_dim, img_dim, sem_dim,
                 gate_dim, que_dims):
        super(CrossGCNLayer, self).__init__()
        self.config = config


        # cross-visual attention
        self.cross_img_att_fact_proj = nn.Linear(in_dims+que_dims, img_att_proj_dim)
        self.cross_img_att_img_proj = nn.Linear(img_dim, img_att_proj_dim)
        self.img_att_proj = nn.Linear(img_att_proj_dim, 1)

        # cross-semantic attention
        self.cross_sem_att_fact_proj = nn.Linear(in_dims+que_dims, sem_att_proj_dim)
        self.cross_sem_att_node_proj = nn.Linear(sem_dim, sem_att_proj_dim)
        self.sem_att_proj = nn.Linear(sem_att_proj_dim, 1)

        # gate
        self.gate_dim = gate_dim
        self.img_gate_fc = nn.Linear(img_dim, gate_dim)
        self.sem_gate_fc = nn.Linear(sem_dim, gate_dim)
        self.fact_gate_fc = nn.Linear(in_dims, gate_dim)
        self.gate_fc = nn.Linear(3 * gate_dim, 3 * gate_dim)
        self.out_fc = nn.Linear(3 * gate_dim, out_dims)

    def forward(self, fact_graph, img_graph, sem_graph, que_embed):
        self.img_graph = img_graph
        self.fact_graph = fact_graph
        self.sem_graph = sem_graph
        self.que_embed = que_embed

        fact_graph.apply_nodes(func=self.apply_node)
        return fact_graph

    
    def apply_node(self, nodes):
        
        node_features = torch.cat([nodes.data['h'], self.que_embed.unsqueeze(0).repeat(self.fact_graph.number_of_nodes(), 1)], dim=1)

        
        img_features = self.img_graph.ndata['h']  
        img_proj = self.cross_img_att_img_proj(img_features)  
        node_proj = self.cross_img_att_fact_proj(node_features)  
        node_proj = node_proj.unsqueeze(1).repeat(1, 36, 1)  
        img_proj = img_proj.unsqueeze(0).repeat(self.fact_graph.number_of_nodes(), 1, 1)
        node_img_proj = torch.tanh(node_proj + img_proj)  
        img_att_value = self.img_att_proj(node_img_proj).squeeze() 
        img_att_value = F.softmax(img_att_value, dim=1)  
        img = torch.matmul(img_att_value, img_features)  



        sem_features = self.sem_graph.ndata['h']  
        sem_num_nodes = self.sem_graph.number_of_nodes()
        sem_proj = self.cross_sem_att_node_proj(sem_features) 
        node_proj = self.cross_sem_att_fact_proj(node_features)  
        node_proj = node_proj.unsqueeze(0).repeat(1, sem_num_nodes, 1)  
        sem_proj = sem_proj.unsqueeze(0).repeat(self.fact_graph.number_of_nodes(), 1, 1)
        node_sem_proj = torch.tanh(node_proj + sem_proj)  
        sem_att_value = self.sem_att_proj(node_sem_proj).squeeze()  
        sem_att_value = F.softmax(sem_att_value, dim=1)  
        sem = torch.matmul(sem_att_value, sem_features)  


        img = self.img_gate_fc(img)
        sem = self.sem_gate_fc(sem)
        fact = self.fact_gate_fc(nodes.data['h'])

        gate = torch.sigmoid(self.gate_fc(torch.cat([fact, img, sem], dim=1)))
        

        h = self.out_fc(gate * torch.cat([fact, img, sem], dim=1))
        return {'h': h}


    def reduce(self, nodes):
        neigh_msg = torch.mean(nodes.mailbox['m'], dim=1) 
       

        h = nodes.data['h']  
        h = torch.cat([neigh_msg, h], dim=1) 
        h = nodes.data['att'] * F.relu(self.apply_fc(h))  #

        # 三种信息 gate
        img = self.img_gate_fc(nodes.data['img'])
        sem = self.sem_gate_fc(nodes.data['sem'])
        fact = self.fact_gate_fc(h)

        gate = torch.sigmoid(self.gate_fc(torch.cat([fact, img, sem], dim=1)))

 
        h = torch.relu(self.out_fc(gate * torch.cat([fact, img, sem], dim=1)))

        return {'h': h}
