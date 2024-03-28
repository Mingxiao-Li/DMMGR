import numpy as np


def parse_reason_tuple_to_graph(reason_tuple):
    entity = []
    entity_connect = []
    edges = []
    edges_connect = []

    multi_reason = False
    for e in reason_tuple:
        if isinstance(e, tuple):
            multi_reason = True
    if not multi_reason:
        if reason_tuple[0] == "Qe":
            e_1 = reason_tuple[1]
            r = reason_tuple[2]
            entity.append(e_1)
            entity.append([reason_tuple[0]])
            entity_connect.append(r)
            entity_connect.append(["None"])
            edges.append(r)
            edges_connect.append([reason_tuple[0]])

        elif reason_tuple[0] == "Qr":
            e_1 = reason_tuple[1]
            e_2 = reason_tuple[2]
            entity.append(e_1)
            entity.append(e_2)
            entity_connect.append([reason_tuple[0]])
            entity_connect.append([reason_tuple[0]])
            edges.append([reason_tuple[0]])
            edges_connect.append(["None"])
    else:
        if reason_tuple[0] == "Qr":
            edges.append([reason_tuple[0]])
            edges_connect.append(["None"])
            for e in reason_tuple[1:]:
                if not isinstance(e,tuple):
                    entity.append(e)
                    entity_connect.append([reason_tuple[0]])
                else:
                    e_obj = e[0]
                    e_2 = e[1]
                    e_r = e[2]
                    entity.append(e_obj)
                    entity_connect.append([reason_tuple[0]])
                    entity.append(e_2)
                    entity_connect.append(e_r)
                    edges.append(e_r)
                    edges_connect.append(e_obj)

        elif reason_tuple[0] == "Qe":
            entity.append([reason_tuple[0]])
            entity_connect.append(["None"])
            for e in reason_tuple[1:]:
                if not isinstance(e,tuple):
                    edges.append(e)
                    edges_connect.append([reason_tuple[0]])
                    rel = e
            for e in reason_tuple[1:]:
                if isinstance(e,tuple):
                    e_obj = e[0]
                    e_2 = e[1]
                    e_r = e[2]
                    entity.append(e_obj)
                    entity_connect.append(rel)
                    entity.append(e_2)
                    entity_connect.append(e_r)
                    edges.append(e_r)
                    edges_connect.append(e_obj)
    total_nodes = entity + edges
    total_connect = entity_connect + edges_connect
    node_type = [1]*len(entity) + [2]*len(edges)  # 1: nodes, 2: edges
    total_connect_id = []

    for c in total_connect:
        if c != ["None"]:
            total_connect_id.append(total_nodes.index(c))
        else:
            total_connect_id.append(-1)
    return total_nodes, total_connect_id, node_type #, entity, entity_connect, edges, edges_connect


def get_reason_graph(config, data, kg_r_to_lang):

    if config.setup == "gt":
        reason = data["reason"]
        qtype = data["qtype"]
        kb = data["KB"]
        reason_tuple = query_parse(qtype,reason,kb,kg_r_to_lang)
        # parse reason flow graph
        total_nodes, total_connect, node_type = parse_reason_tuple_to_graph(reason_tuple)
    elif config.setup == "predict":
        pass
    elif config.setup == "load":
        pass
    return total_nodes, total_connect, node_type


def reason_tuple_to_string_list(reason_tuple):
    reason_string = "( "
    def list_to_string(obj_list):
        s = " [ "
        if "_" in obj_list[0]:
            obj_str = " ".join(obj_list[0].split("_"))
        else:
            obj_str = obj_list[0]
        return s + obj_str+" ] "

    def tuple_to_string(reason_tuple):
        s = "( "
        for element in reason_tuple:
            if isinstance(element, str):
                s += element
            elif isinstance(element,list):
                s+= list_to_string(element)
        s += " )"
        return s

    for element in reason_tuple:
        if isinstance(element, str):
            reason_string += element
        elif isinstance(element,list):
            reason_string += list_to_string(element)
        elif isinstance(element, tuple):
            reason_string += tuple_to_string(element)
    return reason_string.split()


def parse_reason(reason, kg_rel_lang):
    if reason in kg_rel_lang.keys():
        r_word = kg_rel_lang[reason]
    else:
        if "_" in reason:
            r_word = " ".join(reason.split("_"))
        else:
            r_word = reason
    return r_word


def query_parse(qtype, reason, kb, kg_r_to_lang):
    if not kb:
        a, b = reason[0]['e1_label'].lower(), reason[0]['e2_label'].lower()
        if qtype == 0:
            return ('Qr', [a], [b])
        r1 = parse_reason(reason[0]["r"], kg_r_to_lang)
        if qtype == 1:
            return ('Qe', [a], [r1])
        if qtype == 2:
            return ('Qe', [b], [r1])

        c = reason[1]['e2_label'].lower()
        r2 = parse_reason(reason[1]['r'], kg_r_to_lang)
        if qtype == 3:
            return ('Qr', (['object'], [a], [r1]), [c])
        if qtype == 4:
            return ('Qr', [a], (['object'],[c],[r2]))
        if qtype == 5:
            return ('Qe', (['object'], [a], [r1]), [r2])
        if qtype == 6:
            return ('Qe', [r1], (['object'], [c],[r2]))
    else:
        a, b = reason[0]['e1_label'].lower(), reason[0]['e2_label'].lower()
        r1 = parse_reason(reason[0]['r'],kg_r_to_lang)
        if qtype == 2:
            return ('Qe',[b], [r1])

        c = reason[1]['e2_label'].lower()
        r2 = parse_reason(reason[1]['r'],kg_r_to_lang)
        if qtype == 3:
            return ('Qr', (['object'], [a], [r1]), [c])
        if qtype == 4:
            return ('Qr', [a], (['object'], [c], [r2]))
        if qtype == 5:
            return ('Qe', (['object'], [a], [r1]), [r2])
        if qtype == 6:
            return ('Qe', [r1], (['object'], [c], [r2]))


def get_knowledge_graph(config,data, kg_r_to_lang):
    if config.setup == "gt":
        for ele in data["reason"]:
            if "KB" in ele.keys():
                kb = ele
        e_1 = kb["e1_label"].split()
        e_2 = kb["e2_label"].split()
        r = kg_r_to_lang[kb["r"]].split()
        triplets = [[e_1,e_2,r]]
    elif config.setup == "extract":
        pass
    elif config.setup == "load":
        pass
    entity = []
    e1_ids_list = []
    e2_ids_list = []
    edges = []
    for triple in triplets:
        e_1,e_2,r = triple
        if e_1 not in entity:
            entity.append(e_1)
        if e_2 not in entity:
            entity.append(e_2)
        e1_ids_list.append(entity.index(e_1))
        e2_ids_list.append(entity.index(e_2))
        edges.append(r)
    return entity, e1_ids_list, e2_ids_list, edges


def get_img_connection(boxes, num_boxes, k_nearest_node):
    def get_distance(position1, position2):
        return np.linalg.norm(position1-position2)

    x1 = boxes[:, 1]
    y1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    connections = []
    node1_ids_list = []
    node2_ids_list = []
    for i in range(num_boxes):
        node1 = np.array([x_center[i], y_center[i]])
        distance = []
        for j in range(num_boxes):
            node2 = np.array([x_center[j], y_center[j]])
            distance.append(get_distance(node1, node2))
        sorted_index = np.argsort(np.array(distance))
        node1_ids_list.extend([i]*k_nearest_node)
        node2_ids_list.extend(list(sorted_index)[1:k_nearest_node + 1])
    return node1_ids_list,node2_ids_list
