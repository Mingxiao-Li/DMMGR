import json
from typing import Union

def get_word_from_reason_elabel(e_label):
    if "_" in e_label:
        words = e_label.split("_")
    elif " " in e_label:
        words = e_label.split(" ")
    else:
        words = [e_label]
    return words

def get_word_count(data_source, kg_to_lang):
    with open(kg_to_lang, "r") as f:
        kg_to_lang = json.load(f)

    with open(data_path,"r") as f:
        data_source = json.load(f)

    word_count_dict = dict()
    for sample in data_source:
        words = []
        reason = sample["reason"][0]
       # print(sample)
        words.extend(get_word_from_reason_elabel(reason["e1_label"]))
        words.extend(get_word_from_reason_elabel(reason["e2_label"]))
        reason_word = reason["r"]
        if reason_word in kg_to_lang.keys():
            r_word = kg_to_lang[reason_word].split()
        else:
            if "_" in reason_word:
                r_word = reason_word.split("_")
            else:
                r_word = [reason_word]
        words.extend(r_word)
        words.extend(sample["question"].split())
        for word in words:
            if word in word_count_dict:
                word_count_dict[word] += 1
            else:
                word_count_dict[word] = 1
    return word_count_dict

if __name__ == "__main__":
    data_path = "/cw/liir/NoCsBack/testliir/datasets/KR_VQR/question_answer_reason.json"
    kg_lang_path = "/cw/liir/NoCsBack/testliir/datasets/KR_VQR/knowledge_triplet_language.json"
    output_dir = "/cw/liir/NoCsBack/testliir/datasets/KR_VQR/reason_data_triplet_word_count.json"
    wc = get_word_count(data_path,kg_lang_path)
    with open(output_dir,"w") as f:
        json.dump(wc,f)
    print("Done")