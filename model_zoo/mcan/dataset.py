import torch, os, json
import numpy as np
from typing import Union
from x.common.vocabulary import Vocabulary
from x.core.dataset import XDataset
from x.common.image_feature_reader import ImageFeaturesH5Reader
from x.core.registry import registry
from x.common.util import get_word_count
import pdb 

@registry.register_dataset(name="MCANDataset")
class MCANDataset(XDataset):
    # This dataset is for experiment when gt knowledge info (surface) are provided
    def __init__(self, _C):
        super(MCANDataset, self).__init__(_C)
        self.config = _C
        self.top_k_kg = _C.top_k_kg
        self.load_retrieval_kg = _C.load_retrieval_kg
        candidate_answer_path = os.path.join(_C.parent_path, _C.candidate_answer_path)
        with open(candidate_answer_path, "r") as f:
            self.candidate_answers = json.load(f)["answers"]

        # build vocabulary
        if _C.word_count_path is None:
            word_count = get_word_count(self._data)
        else:
            word_count = get_word_count(
                os.path.join(_C.parent_path, _C.word_count_path)
            )

        self._vocabulary = Vocabulary(word_count)

        # set image feature reader
        self.image_feature_reader = ImageFeaturesH5Reader(_C.img_feature_path)
        #self._data = self.filter_datset()

        ########experiment on BU36 features############
        # self.feat_path = "/export/home2/NoCsBack/hci/mingxiao/feats/BU36/"
        ###############################################
        all_kg_path = os.path.join(_C.parent_path, _C.kg_path)
        with open(all_kg_path, "r") as f:
            self.all_kg = json.load(f)["facts"]

        if self.load_retrieval_kg:
            retrieval_kg_path = os.path.join(
                _C.parent_path, _C.retrieval_kg_path
            )
            with open(retrieval_kg_path, "r") as f:
                self.retrieval_kg = json.load(f)

        # get kg_to_lang json
        kg_to_lang_path = os.path.join(_C.parent_path, _C.kg_to_lang_path)
        with open(kg_to_lang_path, "r") as f:
            self.kg_to_lang = json.load(f)

    def encode_img(
        self, features: np.array, num_boxes: int, boxes: np.array, max_region: int = 37
    ):
        num_boxes = min(int(num_boxes), max_region)
        mix_boxes_pad = np.zeros((max_region, boxes.shape[-1]))
        mix_feature_pad = np.zeros((max_region, features.shape[-1]))

        mix_boxes_pad[:num_boxes] = boxes[:num_boxes]
        mix_feature_pad[:num_boxes] = features[:num_boxes]

        features = torch.tensor(mix_feature_pad, dtype=torch.float32)
        img_loc = torch.tensor(mix_boxes_pad, dtype=torch.float32)
        return features, img_loc

    def __getitem__(self, index):
        current_data = self._data[index]
        item = {}
        img_id = current_data["image_id"]
        question = current_data["question"]
        answer = current_data["answer"]
        question_list = self._vocabulary.to_indices(question.split())
        question_tensor, question_mask = self.pad_sequence(
            question_list, max_seq_len=self._config.max_seq_len, return_mask=True
        )

        # if self.split == "train":
        #    answer_tensor = torch.zeros(len(self.candidate_answers),dtype=torch.double)
        #    answer_tensor[self.candidate_answers.index(answer)] = 1
        # else:
        answer_tensor = torch.tensor(self.candidate_answers.index(answer))
        facts_list = []
        kg_list_id = self.retrieval_kg[str(current_data["question_id"])][0]
        for triplet_id in kg_list_id[: self.top_k_kg]:
            kg = self.all_kg[triplet_id]
            e_1, e_2, r = kg["e1_label"], kg["e2_label"], kg["r"]
            r = self.kg_to_lang[r]
            facts = e_1.split(" ") + r.split(" ") + e_2.split(" ") + ["</S>"]
            facts_list += facts
        
        fact_list_ids = self._vocabulary.to_indices(facts_list)
  
        kg_tensor, kg_mask = self.pad_sequence(
            fact_list_ids, max_seq_len=self.config.facts_len,return_mask=True
        )
        item["facts"] = kg_tensor.squeeze()
        item["facts_mask"] = kg_mask.squeeze()

        # get knowledge surface
        #if self._config.use_kb:
        #    assert current_data["KB"] == 1
        #    kb_info = current_data["reason"]
        #    for ele in kb_info:
        #        if "KB" in ele.keys():
        #            kg = ele
        #    head = kg["e1_label"]
        #   tail = kg["e2_label"]
        #    rel = self.kg_to_lang[kg["r"]]
        #    facts = head.split(" ") + rel.split(" ") + tail.split(" ")
        #    kg_list = self._vocabulary.to_indices(facts)
        #    kg_tensor, kg_mask = self.pad_sequence(
        #        kg_list, max_seq_len=self._config.facts_len, return_mask=True
        #    )
        #    item["facts"] = kg_tensor.squeeze()
        #    item["facts_mask"] = kg_mask.squeeze()

        # get image feature
        features, num_boxes, boxes, _, _ = self.image_feature_reader[img_id]
        features, img_loc = self.encode_img(
            features, num_boxes, boxes, max_region=self._config.max_region
        )
        # features = torch.from_numpy(np.load(self.feat_path + str(img_id) + ".jpg.npy", encoding='latin1').astype(np.float32))
        item["question"] = question_tensor.squeeze()
        item["question_mask"] = question_mask.squeeze()
        item["answer"] = answer_tensor
        item["feature"] = features
        item["img_loc"] = img_loc  # do not used during training, thus could be anything
        return item

    def filter_datset(self):
        print("Filtering ......")
        _data = []
        if self._config.only_kb_related:
            for data in self._data:
                if data["KB"] == 1:
                    _data.append(data)

        if self._config.only_kb_not_related:
            for data in self._data:
                if data["KB"] == 0:
                    _data.append(data)

        if self._config.only_q_type is not None:
            for data in self._data:
                if data["qtype"] not in self._config.only_q_type:
                    _data.append(data)
        del self._data
        return _data


if __name__ == "__main__":
    pass
