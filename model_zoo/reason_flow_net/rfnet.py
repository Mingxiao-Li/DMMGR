import torch
import torch.nn as nn
import dgl
from model_zoo.reason_flow_net.modules import Find, \
    Reason, NodeTrans, EdgeTrans, LanguageEncoder
from x import registry


@registry.register_model(name = "RFNet")
class RFNet(nn.Module):

    def __init__(self, config, pretrained_word_embd = None):
        super(RFNet,self).__init__()
        self.config = config
        self.language_encoder = LanguageEncoder(vocab_size=config.Language.vocab_size,
                                                embeded_size=config.Language.embeded_size,
                                                input_size=config.Language.input_size,
                                                hidden_size=config.Language.hidden_size,
                                                num_layers=config.Language.num_layers,
                                                dropout=config.Language.dropout,
                                                rnn_type=config.Language.rnn_type,
                                                bidirectional=config.Language.bidirectional,
                                                glove_word_emebd=pretrained_word_embd)

        # proj handle img feat (node feat and edge feat)
        self.img_fc = nn.Linear(config.img_size, config.hidden_size)
        self.img_loc_fc = nn.Linear(config.loc_size, config.hidden_size)
        self.img_rel_fc = nn.Linear(config.hidden_size*2, config.hidden_size)

        self.fing_node = Find(query_size=config.query_size,
                              key_size=config.key_node_size,
                              mlp_mid_size=config.mlp_mid_size,
                              hidden_size=config.hidden_size)

        self.find_edge = Find(query_size = config.query_edge_size,
                              key_size = config.key_edge_size,
                              mlp_mid_size = config.mlp_mid_size,
                              hidden_size = config.hidden_size)

        self.edge_trans = EdgeTrans()

        self.node_trans = NodeTrans(mid_size = config.mlp_mid_size,
                                    hidden_size = config.hidden_size)


        self.reason_out = Reason(hidden_size = config.hidden_size,
                                      mid_size = config.mlp_mid_size,
                                      num_class = config.num_class)


    def forward(self, img_feat, img_loc, img_node1_id_list,img_node2_id_list,
                kg_entity, kg_e1_ids_list,kg_e2_ids_list,kg_edge,
                r_nodes, r_connects, r_type):


        batch_size, num_obj,_ = img_feat.shape
        _, num_edges = img_node1_id_list.shape

        # img graph node feat
        img_feat = self.img_fc(img_feat)
        img_loc = self.img_loc_fc(img_loc)
        img_node_feat = img_feat+img_loc

        # img graph edge feat
        img_edge_feat = torch.ones(batch_size, num_edges, img_feat.shape[-1]*2)
        for b in range(batch_size):
            node_1_feat = img_feat[b][img_node1_id_list[b]]
            node_2_feat = img_feat[b][img_node2_id_list[b]]
            img_edge_feat[b] = torch.cat([node_1_feat,node_2_feat],dim=1)
        img_edge_feat = self.img_rel_fc(img_edge_feat)

        # build img_graph
        img_graphs = []
        for b in range(batch_size):
            g = dgl.DGLGraph()
            g.add_nodes(num_obj)
            g.add_edge(img_node1_id_list[b],img_node2_id_list[b])
            g.ndata["feat"] = img_node_feat[b]
            g.edges["feat"] = img_edge_feat[b]

        img_graphs.append(g)


        # kg graph node feat
        _, num_kg, entity_len = kg_entity.shape
        kg_entity_len = torch.sum(kg_entity != 1, dim=2)
        kg_node_feat = self.language_encoder(kg_entity, kg_entity_len)
        kg_node_mask = torch.sum(kg_entity==1,dim=2) != entity_len

        # kg graph edge feat
        kg_edge_len = torch.sum(kg_edge != 1, dim=2)
        kg_edge_feat = self.language_encoder(kg_edge, kg_edge_len)

        # build kg_graph
        kg_graphs = []
        for b in range(batch_size):
            g = dgl.DGLGraph()
            g.add_nodes(num_kg)
            g.add_edge(kg_e1_ids_list[b],kg_e2_ids_list[b])

            g.ndata["feat"] = kg_node_feat[b]
            g.ndata["mask"] = kg_node_mask[b]
            g.edges["feat"] = kg_edge_feat[b]
        kg_graphs.append(g)

        # reasoning node feat
        r_nodes_len = torch.sum(r_nodes != 1, dim=2)
        r_node_feat = self.language_encoder(r_nodes, r_nodes_len)
        num_node = torch.sum(r_type != 0, dim=2)

        # build_reasoning graph
        reason_graphs = []
        for b in range(batch_size):
            g = dgl.DGLGraph()
            g.add_nodes(num_node[b])
            for i,n in enumerate(num_node[b]):
                for j in r_connects[i]:
                    if j != -1:
                        g.add_edge(n,j)
            g.ndata["feat"] = r_node_feat[b]
            g.ndata["type"] = r_type[b]
            g.ndata["all_map"] = torch.zeros(num_node[b], num_obj+num_kg)
            g.ndata["img_map"] = torch.zeros(num_node[b], num_obj)
            g.ndata["kg_map"] = torch.zeros(num_node[b], num_kg)

        reason_graphs.append(g)

        batch_output = torch.zeros(batch_size, self.config.num_class)
        # reason process
        for bs in range(batch_size):
            r_g = reason_graphs[bs]
            img_graph = img_graphs[bs]
            kg_graph = kg_graphs[bs]
            for node in dgl.topological_nodes_generator(r_g):
                # reason process
                #  o->o<-o<-<-o edge are nodes with type 2 in r_g
                if len(node) > 1: # two leaves
                    node_1, node_2 = node
                    self.updata_leaf(r_g,node_1,kg_graph,img_graph)
                    self.updata_leaf(r_g,node_2,kg_graph,img_graph)
                else:
                    node_id = node.item()
                    if r_g.out_degree(node_id) > 0 and len(r_g.in_edges(node_id)[0]) == 0: # leaf
                        self.updata_leaf(r_g,node_id,kg_graph,img_graph)
                    elif r_g.out_degree(node_id) > 0 and len(r_g.in_edges(node_id)[0]) > 0: # mid node
                        self.update_node(r_g,node_id,kg_graph,img_graph)
                    else:
                        # reaon output
                        pre_node_ids = r_g.in_edges(node_id)
                        if g.ndata["type"][node_id] == 1:
                            self.update_node(r_g,node_id,kg_graph,img_graph)
                        pred = self.reason_out(r_g,pre_node_ids)
                        batch_output[bs] = pred
        return batch_output

    def updata_leaf(self,r_g,node_id,kg_graph,img_graph):
        img_map, kg_map, all_map, node_re = self.find_node(query=r_g.ndata["feat"][node_id],
                                                           img=img_graph.ndata["feat"],
                                                           kg=kg_graph.ndata["feat"],
                                                           img_mask=None,
                                                           kg_mask=kg_graph.ndata["mask"])
        r_g.ndata["img_map"][node_id] = img_map
        r_g.ndata["kg_map"][node_id] = kg_map
        r_g.ndata["all_map"][node_id] = all_map
        r_g.ndata["feat"][node_id] = node_re

    def update_node(self,r_g,node_id,kg_graph,img_graph):
        if r_g.ndata["type"][node_id] == 2: # edge
            img_edge_map, kg_edge_map, _, _ = self.find_edge(query=r_g.ndata["feat"][node_id],
                                                                              img=img_graph.edata["feat"],
                                                                              kg=kg_graph.edata["feat"],
                                                                              img_mask=None,
                                                                              kg_mask=None)

            pre_node_id = r_g.in_edges(node_id)[0].item()
            self.edge_trans(r_g, img_edge_map, kg_edge_map, img_graph, kg_graph, node_id, pre_node_id)

        elif r_g.ndata["type"][node_id] == 1: # node
            img_map,kg_map,all_map,node_re = self.find_node(query=r_g.ndata["feat"][node_id],
                                                            img=img_graph.ndata["feat"],
                                                            kg=kg_graph.ndata["feat"],
                                                            img_mask=None,
                                                            kg_mask=kg_graph.ndata["mask"])
            r_g.ndata["img_map"][node_id] = img_map
            r_g.ndata["kg_map"][node_id] = kg_map
            r_g.ndata["all_map"][node_id] = all_map
            r_g.ndata["feat"][node_id] = node_re
            pre_node_id = r_g.in_edges(node_id)[0].item()
            self.node_trans(r_g, node_id,pre_node_id,img_graph,kg_graph)


