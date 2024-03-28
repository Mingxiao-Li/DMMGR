import os
import json
from x.core.dataset import XDataset
from x.core.registry import registry
from x.common.vocabulary import Vocabulary

@registry.register_dataset(name="QueryDataset")
class QueyDataset(XDataset):

    def __init__(self, config):
        super(QueyDataset, self).__init__(config)
        self.config = config
        rel_to_lang_path = os.path.join(config.parent_path, config.rel_to_lang_path)
        with open(rel_to_lang_path, "r") as f:
            self.kg_rel_lang = json.load(f)

        with open(os.path.join(config.parent_path,config.word_count_path),"r") as f:
            word_count = json.load(f)
        self.vocabulary = Vocabulary(word_count,min_count=5,special_tokens=["Qr","Qe","(",")","[","]"]) # Qr id 4 Qe id 5

    def __getitem__(self, index):
        current_data = self._data[index]
        question = current_data["question"]
        reason = current_data["reason"]
        qtype = current_data["qtype"]
        kb = current_data["KB"]

        reason_tuple = self.query_parse(qtype, reason, kb)
        reason_str_list = self.reason_tuple_to_string_list(reason_tuple)
        reason_input_str_list= ["<S>"]+reason_str_list
        question_id_list = self.vocabulary.to_indices(question.split())
        reason_input_id_list = self.vocabulary.to_indices(reason_input_str_list)
        reason_gt_list = reason_str_list + ["</S>"]
        reason_gt_id_list = self.vocabulary.to_indices(reason_gt_list)

        question_tensor = self.pad_sequence(question_id_list, max_seq_len=self.config.max_seq_len,return_mask=False)
        reason_in_tensor = self.pad_sequence(reason_input_id_list, max_seq_len=self.config.max_seq_len,return_mask=False)
        reason_gt_tensor = self.pad_sequence(reason_gt_id_list, max_seq_len=self.config.max_seq_len,return_mask=False)
        item = {}
        item["question"] = question_tensor.squeeze(0)
        item["reason_in"] = reason_in_tensor.squeeze(0)
        item["reason_gt"] = reason_gt_tensor.squeeze(0)

        return item

    def reason_tuple_to_string_list(self,reason_tuple):
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

    def parse_reason(self,reason):
        if reason in self.kg_rel_lang.keys():
            r_word = self.kg_rel_lang[reason]
        else:
            if "_" in reason:
                r_word = " ".join(reason.split("_"))
            else:
                r_word = reason
        return r_word

    def query_parse(self,qtype, reason, kb):
        if not kb:
            a, b = reason[0]['e1_label'].lower(), reason[0]['e2_label'].lower()
            if qtype == 0:
                return ('Qr', [a], [b])
            r1 = self.parse_reason(reason[0]["r"])
            if qtype == 1:
                return ('Qe', [a], [r1])
            if qtype == 2:
                return ('Qe', [b], [r1])

            c = reason[1]['e2_label'].lower()
            r2 = self.parse_reason(reason[1]['r'])
            if qtype == 3:
                return ('Qr', (['object'], [a], [r1]), [c])
            if qtype == 4:
                return ('Qr', [a], (['object'], [c], [r2]))
            if qtype == 5:
                return ('Qe', (['object'], [a], [r1]), [r2])
            if qtype == 6:
                return ('Qe', [r1], (['object'], [c], [r2]))
        else:
            a, b = reason[0]['e1_label'].lower(), reason[0]['e2_label'].lower()
            r1 = self.parse_reason(reason[0]['r'])
            if qtype == 2:
                return ('Qe',[b],[r1])

            c = reason[1]['e2_label'].lower()
            r2 = self.parse_reason(reason[1]['r'])
            if qtype == 3:
                return ('Qr', (['object'], [a], [r1]), [c])
            if qtype == 4:
                return ('Qr', [a], (['object'], [c], [r2]))
            if qtype == 5:
                return ('Qe', (['object'], [a], [r1]), [r2])
            if qtype == 6:
                return ('Qe', [r1], (['object'],[c],[r2]))