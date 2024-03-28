import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
import pdb


class GAT(nn.Module):
    def __init__(
        self, in_dim, rel_dim, hidden_dim, out_dim, num_heads, layer_type="GAT"
    ):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(
            in_dim=in_dim,
            rel_dim=rel_dim,
            out_dim=hidden_dim,
            num_heads=num_heads,
            layer_type=layer_type,
        )
        self.layer2 = MultiHeadGATLayer(
            in_dim=hidden_dim * num_heads,
            rel_dim=rel_dim,
            out_dim=out_dim,
            num_heads=1,
            layer_type=layer_type,
        )

    def forward(self, g):
        h = g.ndata["feat"].clone().detach()
        h = self.layer1(g, h)
        h = F.elu(h)
        h = self.layer2(g, h)
        g.ndata["feat"] = h
        return g


class LangGAT(nn.Module):
    def __init__(self, config):
        super(LangGAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(
            in_dim=config.in_dim,
            rel_dim=None,
            out_dim=config.out_dim,
            num_heads=config.num_heads,
            layer_type="GAT",
        )
        self.heads_fc = nn.Linear(config.out_dim * config.num_heads, config.out_dim)

    def forward(self, g):
        h = g.ndata["feat"].clone().detach()
        h = self.layer1(g, h)
        h = F.elu(self.heads_fc(h))
        g.ndata["feat"] = h
        return g


class ProbabilityGraph(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(ProbabilityGraph, self).__init__(**kwargs)

    def edge_prob_update(self, edges):
        prob = edges.src["att"] * edges.dst["att"] + edges.data["att"]
        prob = F.softmax(prob,dim=1)
        return {"att": prob}

    def message_func(self, edges):
        return {"att": edges.data["att"], "src_att": edges.src["att"]}

    def reduce_func(self, nodes):
        prob = nodes.mailbox["att"] * nodes.mailbox["src_att"]
        prob = torch.sum(prob,dim=1) + nodes.data["att"]
        prob = F.softmax(prob, dim=1)
        return {"att": prob}

    def forward(self, pro_graph):
        pro_graph.update_all(self.message_func, self.reduce_func)
        pro_graph.apply_edges(self.edge_prob_update)
        return pro_graph


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, rel_dim, out_dim, num_heads, layer_type, merge="cat"):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            if layer_type == "GAT":
                self.heads.append(GATLayer(in_dim, out_dim))
            elif layer_type == "GQAT":
                self.heads.append(GQATTLayer(in_dim, rel_dim, out_dim))
        self.merge = merge

    def forward(self, g, h):
        head_outs = [attn_head(g, h) for attn_head in self.heads]
        if self.merge == "cat":
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src["feat"], edges.dst["feat"]], dim=1)
        a = self.attn_fc(z2)
        return {"e": F.leaky_relu(a)}

    def message_func(self, edges):
        return {"feat": edges.src["feat"], "e": edges.data["e"]}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox["e"], dim=1)
        h = torch.sum(alpha * nodes.mailbox["feat"], dim=1)
        return {"feat": h}

    def forward(self, g, h):
        z = self.fc(h)
        g.ndata["feat"] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        return g.ndata.pop("feat")


class GQATTLayer(nn.Module):
    def __init__(self, in_dim, rel_dim, out_dim):
        super().__init__()
        self.node_fc = nn.Linear(in_dim, out_dim, bias=False)
        self.rel_fc = nn.Linear(rel_dim, out_dim)
        self.apply_fc = nn.Linear(out_dim * 3, out_dim)
        self.register_parameters()

    def register_parameters(self) -> None:
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.node_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.rel_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.apply_fc.weight, gain=gain)

    def message(self, edges):
        z1 = edges.src["att"].unsqueeze(1) * edges.src["feat"]
        z2 = edges.data["att"].unsqueeze(1) * self.rel_fc(edges.data["feat"])
        msg = torch.cat([z1, z2], dim=1)
        return {"msg": msg}

    def reduce(self, nodes):
        msg = torch.sum(nodes.mailbox["msg"], dim=1)
        h = nodes.data["feat"]
        h = torch.cat([msg, h], dim=1)
        h = nodes.data["att"].unsqueeze(1) * F.relu(self.apply_fc(h))
        return {"feat": h}

    def forward(self, g, h):

        h = self.node_fc(h)
        g.ndata["feat"] = h
        g.update_all(message_func=self.message, reduce_func=self.reduce)
        return g.ndata.pop("feat")


class GCN(nn.Module):
    def __init__(self, config):
        super(GCN, self).__init__()

        self.node_fc = nn.Linear(config.in_dim, config.out_dim)
        self.rel_fc = nn.Linear(config.rel_dim, config.out_dim)
        self.apply_fc_node = nn.Linear(config.out_dim * 3, config.out_dim)
        self.apply_fc_edge = nn.Linear(config.out_dim * 3, config.out_dim)

    def apply_node(self, nodes):
        h = self.node_fc(nodes.data["feat"])
        return {"feat": h}

    def apply_edge(self, edges):
        h = self.rel_fc(edges.data["feat"])
        prob_src = edges.src["att"].unsqueeze(1).expand_as(edges.src["feat"])
        prob_dst = edges.dst["att"].unsqueeze(1).expand_as(edges.dst["feat"])
        z1 = prob_src * edges.src["feat"]
        z2 = prob_dst * edges.dst["feat"]
        msg = torch.cat([z1, z2], dim=1)
        h = torch.cat([msg, h], dim=1)
        f = F.relu(self.apply_fc_edge(h))
        prob_e = edges.data["att"].unsqueeze(1).expand_as(f)
        h = prob_e * f
        return {"feat": h}

    def message(self, edges):
        prob_src = edges.src["att"].unsqueeze(1).expand_as(edges.src["feat"])
        prob_data = edges.data["att"].unsqueeze(1).expand_as(edges.data["feat"])
        z1 = prob_src * edges.src["feat"]
        z2 = prob_data * edges.data["feat"]
        msg = torch.cat([z1, z2], dim=1)
        return {"msg": msg}

    def reduce(self, nodes):
        msg = torch.sum(nodes.mailbox["msg"], dim=1)
        h = nodes.data["feat"]
        h = torch.cat([msg, h], dim=1)
        r = F.relu(self.apply_fc_node(h))
        node_prob = nodes.data["att"].unsqueeze(1).expand_as(r)
        h = node_prob * r
        return {"feat": h}

    def forward(self, g):
        g.apply_nodes(func=self.apply_node)
        # g.apply_edges(func=self.apply_edge)
        g.update_all(message_func=self.message, reduce_func=self.reduce)
        return g

class ImgGCN(nn.Module):
    def __init__(self, in_dims, out_dims, rel_dims):
        super(ImgGCN,self).__init__()
        self.node_fc = nn.Linear(in_dims, in_dims)
        self.rel_fc = nn.Linear(rel_dims, rel_dims)
        self.apply_fc = nn.Linear(in_dims+rel_dims+in_dims,out_dims)
    
    def forward(self,g):
        g.apply_nodes(func = self.apply_node)
        g.update_all(message_func=self.message, reduce_func=self.reduce)
        return g

    def apply_node(self,nodes):
        h = self.node_fc(nodes.data["feat"])
        return {"feat": h}
    
    def message(self,edges):
        z1 = edges.src["att"].unsqueeze(1)* edges.src["feat"]
        z2 = edges.data["att"].unsqueeze(1) * self.rel_fc(edges.data["feat"])
        msg = torch.cat([z1,z2],dim=1)
        return {"msg": msg}
    
    def reduce(self,nodes):
        msg = torch.sum(nodes.mailbox["msg"],dim=1)
        h = nodes.data["feat"]
        h = torch.cat([msg,h],dim=1)
        h = nodes.data["att"].unsqueeze(1) * F.relu(self.apply_fc(h))
        return {"feat": h}


class FactGCN(nn.Module):

    def __init__(self, in_dims,rel_dims, out_dims):
        super(FactGCN, self).__init__()
        self.node_fc = nn.Linear(in_dims, in_dims)
        self.rel_fc = nn.Linear(rel_dims, rel_dims)
        self.apply_fc = nn.Linear(in_dims+rel_dims+in_dims, out_dims)
    
    #def apply_edge(self, edges):
    #    z = torch.cat([edges.src["feat"],edges.dst["feat"]],dim=1)
    #    z = self.edge_fc(z)
    #    z = edges.data["feat"] + z
    #    return {"feat": z} 

    def apply_node(self, nodes):
        h = self.node_fc(nodes.data["feat"])
        return {"feat": h}
    
    def message(self, edges):
        z1 = edges.src["att"].unsqueeze(1) * edges.src["feat"]
        z2 = edges.data["att"].unsqueeze(1) * self.rel_fc(edges.data["feat"])
        msg = torch.cat([z1,z2],dim=1)
        return {"msg": msg}
    
    def reduce(self, nodes):
        msg = torch.sum(nodes.mailbox["msg"],dim=1)
        h = nodes.data["feat"]
        h = torch.cat([msg,h],dim=1)
        h = nodes.data["att"].unsqueeze(1) * F.relu(self.apply_fc(h))
        return {"feat":h}
    
    def forward(self,g):
        #g.apply_edges(func=self.apply_edge)
        g.apply_nodes(func=self.apply_node)
        g.update_all(message_func=self.message, reduce_func=self.reduce)
        return g 

