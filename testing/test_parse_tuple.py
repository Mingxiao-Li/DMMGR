from model_zoo.reason_flow_net.utils import parse_reason_tuple_to_graph, query_parse
import json

data_path = "/cw/liir/NoCsBack/testliir/datasets/KR_VQR/question_answer_reason.json"
kg_r_lang = "/cw/liir/NoCsBack/testliir/datasets/KR_VQR/knowledge_triplet_language.json"

if __name__ == "__main__":
    with open(data_path,"r") as f:
        data = json.load(f)
    with open(kg_r_lang,"r") as f:
        kg_r_lang = json.load(f)

    for i,q in enumerate(data):

        reason = q["reason"]
        type = q["qtype"]
        kb = q["KB"]
        reason_tuple = query_parse(qtype=type,reason=reason,kb=kb,kg_r_to_lang=kg_r_lang)
        tolal_node, total_connect, node_type, entity, entity_connect, edge,edge_connect = parse_reason_tuple_to_graph(reason_tuple)
        print(q["question"])
        print(reason_tuple)
        print("total nodes : ")
        print(tolal_node)
        print("type: ")
        print(node_type)
        print("entity :")
        print(entity)
        print("entity connect :")
        print(entity_connect)
        print("edge : ")
        print(edge)
        print("edge connect: ")
        print(edge_connect)
        print("total connect")
        print(total_connect)
        if i == 100:
            break