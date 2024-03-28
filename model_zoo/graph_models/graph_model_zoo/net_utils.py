from posixpath import pardir
import torch.nn as nn
import torch
import torch.nn.functional as F
import dgl
import pdb


activations = {
    'NON': lambda x: x,
    'TANH': torch.tanh,
    'SIGMOID': F.sigmoid,
    'RELU': F.relu,
    'ELU': F.elu,
}

def build_graph(
    batch_size,
    num_nodes,
    num_edges,
    node_feat,
    edge_feat,
    node1_id_list,
    node2_id_list,
    device,
    edge_att_value=None,
    node_att_value=None,
):
    graphs = []
    for b in range(batch_size):
        num_node = num_nodes[b].item()
        num_edge = num_edges[b].item()

        g = dgl.graph(
            (node1_id_list[b][:num_edge], node2_id_list[b][:num_edge]),
            num_nodes=num_node,
        )
        g.to(device)
        if edge_att_value != None:
            g.edata["att"] = edge_att_value[b][:num_edge]
        if node_att_value != None:
            g.ndata["att"] = node_att_value[b][:num_node]
        g.ndata["feat"] = node_feat[b][:num_node]
        g.edata["feat"] = edge_feat[b][:num_edge]
        graphs.append(g)
    graph_batch = dgl.batch(graphs)
    return graph_batch


def get_graph_feat(nodes, edges, encoder, max_node_len, max_edge_len, use_scene_graph):
    num_nodes = torch.sum(torch.sum(nodes, dim=2) != max_node_len, dim=1)
    num_edges = torch.sum(torch.sum(edges, dim=2) != max_edge_len, dim=1)
    node_rel_len = torch.sum(nodes != 1, dim=2)
    node_rel_len += node_rel_len == 0
    node_embedding = encoder(nodes.long())
    node_feat = torch.sum(node_embedding, dim=2)
    node_feat = node_feat / (node_rel_len.unsqueeze(2).expand_as(node_feat))

    if use_scene_graph:
        edge_rel_len = torch.sum(edges != 1, dim=2)
        edge_rel_len += edge_rel_len == 0
        edge_embedding = encoder(edges)
        edge_feat = torch.sum(edge_embedding, dim=2)
        edge_feat = edge_feat / (edge_rel_len.unsqueeze(2).expand_as(edge_feat))
    else:
        edge_feat = torch.tensor(edges)

    return node_feat, edge_feat, num_nodes, num_edges


def build_and_merge_graph(
    batch_size,
    num_img_nodes,
    num_img_edges,
    img_node_feat,
    img_edge_feat,
    img_node1_ids_list,
    img_node2_ids_list,
    num_kg_nodes,
    num_kg_edges,
    kg_node_feat,
    kg_edge_feat,
    kg_node1_ids_list,
    kg_node2_ids_list,
):
    graphs = []
    graphs_info = []
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
            img_node_feat[b][:num_img_node],kg_node_feat[b][:num_kg_node]
        )

        if cosine_sim.shape[0] == 0:
            pdb.set_trace()
        pos = torch.argmax(cosine_sim).item()

        img_id, kg_id = (
            pos // num_kg_node,
            pos - (num_kg_node * (pos // num_kg_node)),
        )
        all_nodes = torch.cat(
            [img_node_feat[b][:num_img_node], kg_node_feat[b][:num_kg_node]],
            dim=0,
        )
        all_edges = torch.cat(
            [img_edge_feat[b][:num_img_edge], kg_edge_feat[b][:num_kg_edge]],
            dim=0
        )
        kg_node1_id_list += num_img_node
        kg_node2_id_list += num_img_node

        # merge node index
        for i in range(num_kg_edge):
            if kg_node2_id_list[i].item() == kg_id + num_img_node:
                kg_node2_id_list[i] = img_id
            if kg_node1_id_list[i].item() == kg_id + num_img_node:
                kg_node1_id_list[i] = img_id
        
        all_node1_ids_list = torch.cat([img_node1_id_list, kg_node1_id_list], dim=0)
        all_node2_ids_list = torch.cat([img_node2_id_list, kg_node2_id_list], dim=0)
        g = dgl.graph(
            (all_node1_ids_list, all_node2_ids_list),
            num_nodes = num_img_node + num_kg_node
        )
        g.ndata["feat"] = all_nodes
        g.edata["feat"] = all_edges 
        graphs.append(g)
        graphs_info.append((all_node1_ids_list,all_node2_ids_list))
    graph_batch = dgl.batch(graphs)
    
    return graph_batch, graphs_info
    


def cosine(input_1, input_2):
    r"""
    compute consine similarity of each pair tensors of input1 and input 2
    input_1 : shape 1,num_obj, dim
    input_2 : shape 1, num_obj2, dim
    """
    score = torch.matmul(input_1, input_2.permute(1, 0))
    input_1_norm = torch.norm(input_1, dim=1)
    input_2_norm = torch.norm(input_2, dim=1)
    norm = torch.matmul(
        input_1_norm.unsqueeze(1), input_2_norm.unsqueeze(1).permute(1, 0)
    )
    cosine_similarity = score / norm
    return cosine_similarity


def generate_bidirection_edges(node1_id_list, node2_id_list):
    pairs = []
    for (h, t) in zip(node1_id_list, node2_id_list):
        if (h.item(), t.item()) not in pairs:
            pairs.append((h.item(), t.item()))
    node1, node2 = zip(*pairs)
    node1_list = list(node1)
    node2_list = list(node2)
    node1_id_list = node1_list + node2_list
    node2_id_list = node2_list + node1_list
    return node1_id_list, node2_id_list




class FocalLoss_Multilabel(nn.Module):
    def __init__(self,class_num, alpha=0.25,  gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss_Multilabel,self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()
        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average
    
    def forward(self,pred,target):
        prob = self.softmax(pred.view(-1, self.class_num))
        prob = prob.clamp(min=0.0001, max=1.0)

        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1,1).long(), 1.)

        if self.use_alpha:
            batch_loss = -self.alpha.double() * torch.pow(1-prob, self.gamma).double()*prob.log()*target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        
        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss
        
