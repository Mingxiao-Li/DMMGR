from typing import Union, List
import os
import numpy as np
import torch, json
import torch.nn as nn
import h5py
import lmdb
import pickle
from tqdm import tqdm


def get_numpy_word_embed(
    word2index: Union[dict, None], pretrained_wordemb_path: str
) -> np.array:
    """
    read pretrained word embedding from file and conver it to np.array

    :param word2index: dict {word:index...}
    :param pretrained_wordemb_path:  txt file （word,num1,num2,...）
           pretraiend_wordemb_path_np: .npy file (word_1_embd, word_2_embd,...)
    :return: pretrained word embedding

    Example
    -------
    >>> word_embedding = get_numpy_word_embed(word2index,pretrained_wordemb_path)
    >>> embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding))
    """
    pretrained_wordemb_path_np = pretrained_wordemb_path.replace("txt", "npy")
    if os.path.exists(pretrained_wordemb_path_np):
        return np.load(pretrained_wordemb_path_np)

    words_embed = {}
    with open(pretrained_wordemb_path, "r") as pretrained_wordemb_file:
        lines = pretrained_wordemb_file.readlines()
        for line in lines:
            line_list = line.split()
            word = line_list[0]
            embed = line_list[1:]
            embed = [float(num) for num in embed]
            words_embed[word] = embed
    dim = len(embed)
    index2word = {index: word for word, index in word2index.items()}
    id2emb = {}
    for index in range(len(word2index)):
        if index2word[index] in words_embed:
            id2emb[index] = words_embed[index2word[index]]
        else:
            id2emb[index] = [0.0] * dim
            # PAD and all special token are initialized to zero tensor

    word_embedings = np.array([id2emb[index] for index in range(len(word2index))])

    # set unk embedding which the averate embedding of all words
    word_embedings[0, :] = np.mean(word_embedings[10:, :], axis=0)
    np.save(pretrained_wordemb_path_np, word_embedings)
    print(
        "New word_embedding with correct index is saved to 'glove.42B.300d.npy' successfully,"
        "You can load this .npy file drectly next time !!"
    )
    return word_embedings


def get_word_count(data_source: Union[str, dict]):
    if isinstance(data_source, str):
        with open(data_source, "r") as f:
            word_count_dict = json.load(f)
        return word_count_dict

    word_count_dict = dict()
    for sample in data_source:
        for word in sample["question"].lower().split():
            if word in word_count_dict:
                word_count_dict[word] += 1
            else:
                word_count_dict[word] = 1
    return word_count_dict


def logging_format(losses: List[str], show_iter: bool = True) -> str:
    # A simple function to return format of showing loss info
    # example : log_f % tuple([])
    if show_iter:
        basic_format = "[Ep: %.2f][Iter: %d]"
    else:
        basic_format = "[EP: %.2f]"
    loss_f = ""
    for loss in losses:
        loss_f += "[{} :%.3g]".format(loss)
    return basic_format + loss_f


def h5py_to_lmdb(h5py_path, lmdb_path, map_size=1099511627776):
    """
    h5py_path: directory of h5py file
    lmdb_path: output directory
    """
    env = lmdb.open(lmdb_path, map_size=map_size)
    txn = env.begin(write=True)
    f = h5py.File(h5py_path, "r")
    for key in tqdm(f.keys()):
        info_dict = json.loads(f.get(str(key))[()].decode("utf-8").replace("'", '"'))
        info = pickle.dumps(info_dict)
        txn.put(key=key.encode(), value=info)
    txn.commit()
    env.close()
