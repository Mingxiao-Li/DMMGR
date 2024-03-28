from networkx.algorithms.assortativity import pairs
import torch
import numpy as np
import pdb
import spacy
import json
import h5py
import time


class ImgSceneGraphGetter:
    def __init__(
        self,
        scene_graphs,
        vocabulary,
        max_num_entity,
        max_entity_len,
        max_num_edge,
        max_edge_len,
        scene_graphs_info=None,
    ):
        self.scene_graphs = scene_graphs
        self.vocabulary = vocabulary
        self.max_num_entity = max_num_entity
        self.max_num_edge = max_num_edge
        self.max_entity_len = max_entity_len
        self.max_edge_len = max_edge_len
        self.use_generated_sg = False

        if scene_graphs_info != None:
            self.use_generated_sg = True
            self.img_id_to_index = scene_graphs_info["img_id_to_index"]
            self.ind_to_classes = scene_graphs_info["ind_to_classes"]
            self.ind_to_predicates = scene_graphs_info["ind_to_predicates"]

    def generate_connections(self, img_index):

        img_index = str(img_index)
        if self.use_generated_sg:
            return self.read_generated_scene_graph(img_index)

        relations = self.scene_graphs[img_index]["relationships"]
        objects = self.scene_graphs[img_index]["objects"]

        entity_list = []
        entity_id_list = []
        edge_list = []
        node1_ids_list = []
        node2_ids_list = []
        visited_entity_pair = []
        for triplet in relations:
            e_1 = objects[str(triplet["subject_id"])]
            e_2 = objects[str(triplet["object_id"])]

            e_1_id = triplet["subject_id"]
            e_2_id = triplet["object_id"]

            e_1_name = " ".join(e_1["names"][0].lower().split("_"))
            e_2_name = " ".join(e_2["names"][0].lower().split("_"))
            r = " ".join(triplet["predicate"].lower().split("_"))

            if e_1_name not in entity_list:
                is_e1_new = True

            elif e_1_name in entity_list:
                boxA = [e_1["x"], e_1["y"], e_1["x"] + e_1["w"], e_1["y"] + e_1["h"]]
                objs = []
                for i, e in enumerate(entity_list):
                    if e == e_1_name:
                        objs.append(objects[str(entity_id_list[i])])
                ious = []
                for obj in objs:
                    boxB = [
                        obj["x"],
                        obj["y"],
                        obj["x"] + obj["w"],
                        obj["y"] + obj["h"],
                    ]
                    ious.append(self.compute_iou(boxA, boxB))
                max_iou = max(ious)
                if max_iou <= 0.1:
                    is_e1_new = True
                else:
                    is_e1_new = False
                    index = ious.index(max_iou)
                    obj = objs[index]
                    e_1_id = obj["object_id"]

            if e_2_name not in entity_list:
                is_e2_new = True
            elif e_2_name in entity_list:
                boxA = [e_2["x"], e_2["y"], e_2["x"] + e_2["w"], e_2["y"] + e_2["h"]]
                objs = []
                for i, e in enumerate(entity_list):
                    if e == e_2_name:
                        objs.append(objects[str(entity_id_list[i])])
                ious = []
                for obj in objs:
                    boxB = [
                        obj["x"],
                        obj["y"],
                        obj["x"] + obj["w"],
                        obj["y"] + obj["h"],
                    ]
                    ious.append(self.compute_iou(boxA, boxB))
                max_iou = max(ious)
                if max_iou <= 0.2:
                    is_e2_new = True
                else:
                    is_e2_new = False
                    index = ious.index(max_iou)
                    obj = objs[index]
                    e_2_id = obj["object_id"]

            if (e_1_id, e_2_id) in visited_entity_pair:
                continue
            else:
                visited_entity_pair.append((e_1_id, e_2_id))

            if is_e1_new:
                entity_list.append(e_1_name)
                entity_id_list.append(e_1_id)

            if is_e2_new:
                entity_list.append(e_2_name)
                entity_id_list.append(e_2_id)

            edge_list.append(r)
            node1_ids_list.append(entity_id_list.index(e_1_id))
            node2_ids_list.append(entity_id_list.index(e_2_id))
            if (
                len(entity_list) >= self.max_num_entity
                or len(edge_list) >= self.max_num_edge
            ):
                break

        # print("obj", len(objects))
        # print("rel", len(relations))
        # print("edge", len(edge_list))
        # print("entity", len(entity_list))
        # print("entity_id", len(entity_id_list))
        return entity_list, edge_list, node1_ids_list, node2_ids_list

    def get_entity_and_edge_list(self, img_index):
        entity_list, edge_list, _, _ = self.generate_connections(img_index)
        return entity_list, edge_list

    def encode_graph_element(self, element_list, num_ele, ele_len):
        tensor_list = torch.ones(num_ele + 1, ele_len, dtype=torch.int64)
        for i, e in enumerate(element_list):
            e_list = e.split()
            e_list_idx = self.vocabulary.to_indices(e_list)
            if len(e_list_idx) > ele_len:
                tensor_list[i, :] = torch.tensor(e_list_idx[:ele_len])
            else:
                tensor_list[i, : len(e_list_idx)] = torch.tensor(e_list_idx)
        return tensor_list

    def get_img_graph(self, img_index):

        (
            entity_list,
            edge_list,
            node1_ids_list,
            node2_ids_list,
        ) = self.generate_connections(img_index)

        entity_list_tensor = self.encode_graph_element(
            entity_list, self.max_num_entity, self.max_entity_len
        )

        edge_list_tensor = self.encode_graph_element(
            edge_list, self.max_num_edge, self.max_edge_len
        )
        node1_ids_list_tensor = torch.ones(self.max_num_edge, dtype=torch.int64)
        node2_ids_list_tensor = torch.ones(self.max_num_edge, dtype=torch.int64)
        if len(node1_ids_list) > self.max_num_edge:
            node1_ids_list_tensor = torch.tensor(node1_ids_list[: self.max_num_edges])
        else:
            node1_ids_list_tensor[: len(node1_ids_list)] = torch.tensor(node1_ids_list)

        if len(node2_ids_list) > self.max_num_edge:
            node2_ids_list_tensor = torch.tensor(node2_ids_list[: self.max_num_edges])
        else:
            node2_ids_list_tensor[: len(node2_ids_list)] = torch.tensor(node2_ids_list)

        return (
            entity_list_tensor,
            edge_list_tensor,
            node1_ids_list_tensor,
            node2_ids_list_tensor,
        )

    def read_generated_scene_graph(self, img_id):

        sg_index = self.img_id_to_index[img_id]
        group = self.scene_graphs.get(str(sg_index))
        bbox = group["bbox"][()]
        bbox_label = group["bbox_labels"][()]

        if self.max_num_entity < len(bbox):
            bbox = bbox[: self.max_num_entity]
            bbox_label = bbox_label[: self.max_num_entity]

        rel_pairs = group["rel_pairs"][()]
        rel_label = group["rel_labels"][()]
        edge_list = []
        node1_ids_list = []
        node2_ids_list = []
        for i, pair in enumerate(rel_pairs):
            box1, box2 = pair
            if box1 > len(bbox) or box2 > len(bbox):
                continue
            edge_list.append(self.ind_to_predicates[rel_label[i]])
            node1_ids_list.append(box1)
            node2_ids_list.append(box2)
        if len(edge_list) >= self.max_num_edge:
            edge_list = edge_list[: self.max_num_edge]
            node1_ids_list = node1_ids_list[: self.max_num_edge]
            node2_ids_list = node2_ids_list[: self.max_num_edge]
        entity_list = [self.ind_to_classes[index] for index in bbox_label]
        return (
            entity_list,
            edge_list,
            node1_ids_list,
            node2_ids_list,
        )

    def compute_iou(self, boxA, boxB):
        x1_boxA, y1_boxA, x2_boxA, y2_boxA = boxA
        x1_boxB, y1_boxB, x2_boxB, y2_boxB = boxB
        x1_union = max(x1_boxA, x1_boxB)
        y1_union = max(y1_boxA, y1_boxB)
        x2_union = min(x2_boxA, x2_boxB)
        y2_union = min(y2_boxA, y2_boxB)

        unionArea = max(0, x2_union - x1_union) * max(0, y2_union - y1_union)
        boxAArea = (x2_boxA - x1_boxA) * (y2_boxA - y1_boxA)
        boxBArea = (x2_boxB - x1_boxB) * (y2_boxB - y1_boxB)
        iou = unionArea / float(boxAArea + boxBArea - unionArea)
        return iou


class KGGraphGetter:
    def __init__(
        self,
        kg_facts,
        retrieval_kg,
        vocabulary,
        rel_to_lang,
        top_k_kg,
        max_entity_len,
        max_edge_len,
    ):
        self.kg_facts = kg_facts
        self.vocabulary = vocabulary
        self.rel_to_lang = rel_to_lang
        self.retrieval_kg = retrieval_kg
        self.top_k_kg = top_k_kg
        self.max_num_entity = top_k_kg * 2
        self.max_entity_len = max_entity_len
        self.max_num_edges = top_k_kg
        self.max_edge_len = max_edge_len

    def generate_connections(self, quesion_id):
        kg_list_id = self.retrieval_kg[str(quesion_id)][0]
        entity_list = []
        edge_list = []
        node1_ids_list = []
        node2_ids_list = []
        edge_list = []
        visited = []

        for triplet_id in kg_list_id[: self.top_k_kg]:
            kg = self.kg_facts[triplet_id]
            e_1, e_2, r = kg["e1_label"], kg["e2_label"], kg["r"]
            if e_1 == "" or e_2 == "":
                continue
            if e_1 == e_2:
                continue
            if (e_1, e_2) in visited or (e_2, e_1) in visited:
                continue
            else:
                visited.append((e_1, e_2))

            if e_1 not in entity_list:
                entity_list.append(e_1)
            if e_2 not in entity_list:
                entity_list.append(e_2)

            node1_ids_list.append(entity_list.index(e_1))
            node2_ids_list.append(entity_list.index(e_2))
            edge_list.append(" ".join(self.rel_to_lang[r].split()))
        return entity_list, edge_list, node1_ids_list, node2_ids_list

    def get_entity_and_edge_list(self, question_id):
        entity_list, edge_list, _, _ = self.generate_connections(question_id)
        return entity_list, edge_list

    def encode_graph_element(self, element_list, num_ele, ele_len):
        tensor_list = torch.ones(num_ele, ele_len, dtype=torch.int64)
        for i, e in enumerate(element_list):
            e_list = e.split()
            e_list_idx = self.vocabulary.to_indices(e_list)
            if len(e_list_idx) > ele_len:
                tensor_list[i, :] = torch.tensor(e_list_idx[:ele_len])
            else:
                tensor_list[i, : len(e_list_idx)] = torch.tensor(e_list_idx)
        return tensor_list

    def get_kg_graph(self, question_id):
        (
            entity_list,
            edge_list,
            node1_ids_list,
            node2_ids_list,
        ) = self.generate_connections(question_id)
        entity_list_tensor = self.encode_graph_element(
            entity_list,
            self.max_num_entity,
            self.max_entity_len,
        )
        edge_list_tensor = self.encode_graph_element(
            edge_list,
            self.max_num_edges,
            self.max_edge_len,
        )

        node1_ids_list_tensor = torch.ones(self.max_num_edges, dtype=torch.int64)
        node2_ids_list_tensor = torch.ones(self.max_num_edges, dtype=torch.int64)
        node1_ids_list_tensor[: len(node1_ids_list)] = torch.tensor(node1_ids_list)
        node2_ids_list_tensor[: len(node2_ids_list)] = torch.tensor(node2_ids_list)

        return (
            entity_list_tensor,
            edge_list_tensor,
            node1_ids_list_tensor,
            node2_ids_list_tensor,
        )


class ImgGraphGetter:
    def __init__(
        self,
        feature_reader,
        max_region,
        k_nearest,
        use_symbolic,
        object_path=None,
        vocabulary=None,
    ):
        self.img_feature_reader = feature_reader
        self.max_region = max_region
        self.k_nearest = k_nearest
        self.use_symbolic = use_symbolic
        self.max_num_edges = max_region * k_nearest
        if use_symbolic:
            assert object_path is not None
            assert vocabulary is not None
            self.vocabulary = vocabulary
            with open(object_path, "r") as f:
                self.objects = json.load(f)

    def build_img_connection(self, boxes, num_boxes):
        def get_distance(position_1, position_2):
            return np.linalg.norm(position_1 - position_2, ord=2)

        rel_num_edges = min(num_boxes, self.k_nearest)
        x1 = boxes[:, 1]
        y1 = boxes[:, 2]
        x2 = boxes[:, 3]
        y2 = boxes[:, 4]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        node1_ids_list = []
        node2_ids_list = []
        for i in range(num_boxes):
            node1 = np.array([x_center[i], y_center[i]])
            distance = []
            for j in range(num_boxes):
                node2 = np.array([x_center[j], y_center[j]])
                distance.append(get_distance(node1, node2))
            sorted_idindex = np.argsort(np.array(distance))
            node1_ids_list.extend([i] * (rel_num_edges - 1))
            node2_ids_list.extend(list(sorted_idindex[1:rel_num_edges]))
        return node1_ids_list, node2_ids_list

    def get_relative_position(self, boxes_ori, node1_ids_list, node2_ids_list):
        assert len(node1_ids_list) == len(node2_ids_list)
        x1 = boxes_ori[:, 1]
        y1 = boxes_ori[:, 2]
        x2 = boxes_ori[:, 3]
        y2 = boxes_ori[:, 4]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        relate_loc = []
        for i in range(len(node1_ids_list)):
            r1_id = node1_ids_list[i]
            r2_id = node2_ids_list[i]
            region1_loc = boxes_ori[r1_id]
            region2_loc = boxes_ori[r2_id]
            # region_loc = [x1,y1,x2,y2,region_area/whole_area]
            w_i = region1_loc[2] - region1_loc[0]
            h_i = region1_loc[3] - region1_loc[1]
            w_j = region2_loc[2] - region2_loc[0]
            h_j = region2_loc[3] - region2_loc[1]
            relate_loc.append(
                [
                    (x_center[r2_id] - x_center[r1_id]) / (w_i * h_i) ** 0.5,
                    (y_center[r2_id] - y_center[r1_id]) / (w_i * h_i) ** 0.5,
                    w_j / w_i,
                    h_j / h_i,
                    w_j * h_j / (w_i * h_i),
                ]
            )
        return torch.tensor(relate_loc)

    def get_img_graph(self, img_id):
        features, num_boxes, boxes, boxes_ori, cls_prob = self.img_feature_reader[
            img_id
        ]

        num_boxes = min(int(num_boxes), self.max_region)
        if not self.use_symbolic:
            mix_feature_pad = np.zeros((self.max_region, features.shape[-1]))
            mix_feature_pad[:num_boxes] = features[:num_boxes]
            img_features = torch.tensor(mix_feature_pad, dtype=torch.float32)
        else:
            pred_obj_id = np.argmax(cls_prob, axis=1)
            img_features = np.ones((self.max_region, 2))
            objs = []
            boxes_ori_ = []
            k = 0
            for i, obj_id in enumerate(pred_obj_id):
                if obj_id == 0:
                    continue
                boxes_ori_.append(boxes_ori[i])
                obj_list = self.objects[str(obj_id)].split()
                objs.append(obj_list)
                obj_ids = self.vocabulary.to_indices(obj_list)
                if len(obj_ids) == 1:
                    img_features[k][0] = obj_ids[0]
                elif len(obj_ids) == 2:
                    img_features[k] = obj_ids
                else:
                    raise ValueError("object id list larget than 2")
                k += 1
            boxes_ori = np.array(boxes_ori_)
            num_boxes, _ = boxes_ori.shape

        node1_ids_list, node2_ids_list = self.build_img_connection(boxes_ori, num_boxes)

        img_loc_tensor = self.get_relative_position(
            boxes_ori, node1_ids_list, node2_ids_list
        )

        img_loc_tensor_pad = torch.ones(self.max_num_edges, 5)
        img_loc_tensor_pad[: img_loc_tensor.shape[0], :] = img_loc_tensor
        node1_ids_list_tensor = torch.ones(self.max_num_edges, dtype=torch.int64)
        node2_ids_list_tensor = torch.ones(self.max_num_edges, dtype=torch.int64)
        node1_ids_list_tensor[: len(node1_ids_list)] = torch.tensor(node1_ids_list)
        node2_ids_list_tensor[: len(node2_ids_list)] = torch.tensor(node2_ids_list)

        assert (
            node1_ids_list_tensor.shape[0]
            == node2_ids_list_tensor.shape[0]
            == img_loc_tensor_pad.shape[0]
        )

        return (
            img_features,
            img_loc_tensor_pad,
            node1_ids_list_tensor,
            node2_ids_list_tensor,
        )


class LanguageGraphGetter:
    def __init__(self, max_num_edge, connections=None) -> None:
        super(LanguageGraphGetter, self)
        self.connections = connections
        self.en_nlp = spacy.load("en_core_web_sm")
        self.max_num_edge = max_num_edge

    def generate_connections(self, question):
        question = " ".join(question.split())
        doc = self.en_nlp(question)
        node_ids_list1 = []
        node_ids_list2 = []
        for token in doc:
            if token.text == token.head.text:
                continue
            node_ids_list1.append(token.head.i)
            node_ids_list2.append(token.i)

        return node_ids_list1, node_ids_list2

    def get_lang_graph(self, question, question_id=None):
        if self.connections is None:
            node_ids_list1, node_ids_list2 = self.generate_connections(question)
        else:
            assert question_id is not None
            node_ids_list1, node_ids_list2 = (
                self.connections[str(question_id)]["node_ids_list1"],
                self.connections[str(question_id)]["node_ids_list2"],
            )
        node_ids_list1_tensor = torch.ones(self.max_num_edge, dtype=torch.int64) * 100
        node_ids_list2_tensor = (
            torch.ones(self.max_num_edge, dtype=torch.int64) * 100
        )  # pad
        node_ids_list1_tensor[: len(node_ids_list1)] = torch.tensor(node_ids_list1)
        node_ids_list2_tensor[: len(node_ids_list2)] = torch.tensor(node_ids_list2)

        return node_ids_list1_tensor, node_ids_list2_tensor


def put_answer_in_graph(
    ans_synsets, answer, img_node_list, img_edge_list, kg_node_list, kg_edge_list
):

    """
    graph_node_edge_order: img_node, kg_node, img_edge, kg_edge
    answer: index of real answer in graph order
    """
    synsets_rel = ans_synsets["relationships"]["synset_to_predicate"]
    synsets_obj = ans_synsets["objects"]["synset_to_name"]
    answer = " ".join(answer.split("_"))
    answers = []
    if answer in synsets_rel:
        answers.extend(list(set(synsets_rel[answer])))
    if answer in synsets_obj:
        answers.extend(list(set(synsets_obj[answer])))
    total_list = img_node_list + kg_node_list + img_edge_list + kg_edge_list
    find_answer = False
    answer_id = 0
    for a in answers:
        if a in total_list:
            answer_id = total_list.index(a)
            find_answer = True
            break
        else:
            continue
    if find_answer != True:
        # print(answers)
        # print(total_list)
        t += 1
        print(t)
    # exit(1)
    answer_ind = torch.tensor(answer_id)
    return answer_ind, t
