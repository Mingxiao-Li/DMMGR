import torch
import torch.nn as nn
import dgl
from model_zoo.reason_flow_net.modules import FindNode, FindEdge, ReasonEdge, ReasonNode, NodeTrans, NodeMatchMap

class Net(nn.Module):

    def __init__(self, config, pretrained_word_embd = None):
        super(Net,self).__init__()

        # proj handle img feat (node feat and edge feat)
        self.img_fc = nn.Linear(config.img_size, config.hidden_size)
        self.img_loc_fc = nn.Linear(config.loc_size, config.hidden_size)
        self.img_rel_fc = nn.Linear(config.hidden_size*2, config.hidden_size)

        self.find_node = FindNode(query_node_size = config.query_node_size,
                                  key_node_size = config.key_node_size,
                                  mlp_mid_size = config.mlp_mid_size,
                                  hidden_size = config.hidden_size)

        self.find_edge = FindEdge(query_edge_size = config.query_edge_size,
                                  key_edge_size = config.key_edge_size,
                                  mlp_mid_size = config.mlp_mid_size,
                                  hidden_size = config.hidden_size)

        self.node_trans = NodeTrans(mid_size = config.mlp_mid_size,
                                    hidden_size = config.hidden_size)

        self.node_match_map = NodeMatchMap(query_size = config.query_size,
                                           hidden_size = config.hidden_size,
                                           node_size = config.node_size,
                                           mid_size = config.mlp_mid_size)

        self.reason_node = ReasonNode(hidden_size = config.hidden_size,
                                      mid_size = config.mlp_mid_size,
                                      num_class = config.num_class)

        self.reason_edge = ReasonEdge(hidden_size = config.hidden_size,
                                      mid_size = config.mlp_mid_size,
                                      num_class = config.num_class)

    def forward(self, img_feat, img_loc, img_connection,kg_entity, kg_connect,kg_r,
                r_nodes, r_connects, r_type):
        # img_connection :dict{1: [nodes connected with node 1]}

        batch_size, num_obj,_ = img_feat.shape

        # img graph node feat
        img_feat = self.img_fc(img_feat)
        img_loc = self.img_loc_fc(img_loc)
        img_node_feat = img_feat+img_loc

        # img graph edge feat
        total_num_edges = num_obj * sum([len(x) for x in img_connection.values()])
        img_edge_feat = torch.ones(batch_size, total_num_edges, img_node_feat.shape[-1])
        k = 0
        for i in range(num_obj):
            for j in range(img_connection[i]):
                d_node_id = img_connection[i][j]
                img_edge_feat[:,k,:] = self.img_rel_fc(torch.cat([img_feat[i],img_feat[d_node_id]]))
                k += 1

        # build img_graph
        img_graph = []
        for b in range(batch_size):
            g = dgl.DGLGraph()
            g.add_nodes(num_obj)
            g.ndata["feat"] = img_node_feat[b,:]
            for i,s in enumerate(range(num_obj)):
                for d in img_connection[i]:
                    g.add_edge(s, d)
            g.edata["feat"] = img_edge_feat[b,:]
            img_graph.append(g)
        img_batch_graph = dgl.batch(img_graph)


        #build kg_graph
        # need modify when kg is more than 1
        kg_graph = []
        for j in range(batch_size):
            g = dgl.DGLGraph()


        #build_reasoning graph
        reason_graphs = []
        for k in range(batch_size):
            reason_graph = reason_graphs[k]
            for node in dgl.topological_nodes_generator(reason_graph):
                # reason process
                #Find node
                node = 2
                # Find edge
                edge = reason_graph.edata["feat"][reason_graph.out_edges(node,"eid")]
                # Trans


        for bs in range(batch_size):
            pass