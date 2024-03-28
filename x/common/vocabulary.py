"""
A Vocabulary maintains a mapping between words and corresponding unique integers, holds special
integers (tokens) for indicating start and end of sequence, and offers functionality to map
out-of-vocabulary words to the corresponding token.
"""

import json
import os
from typing import List, Union, Optional


class Vocabulary(object):

    PAD_TOKEN = "<pad>"
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    UNK_TOKEN = "<unk>"

    UNK_INDEX = 0
    PAD_INDEX = 1
    SOS_INDEX = 2
    EOS_INDEX = 3

    # some space for other special token
    SPECIAL_TOKEN = [
        "special_1",
        "special_2",
        "special_3",
        "special_4",
        "special_5",
        "special_6",
    ]
    SPECIAL_TOKEN_INDEX = [4, 5, 6, 7, 8, 9]

    def __init__(
        self,
        word_counts: Union[str, dict],
        min_count: int = 5,
        special_tokens: Union[None, List[str]] = None,
    ):
        super().__init__()

        if type(word_counts) == str:
            if not os.path.exists(word_counts):
                raise FileNotFoundError(f"Word couts fo not exist at {word_counts}")

            with open(word_counts, "r") as word_count_file:
                word_counts = json.load(word_count_file)

        # form a list of (word, count) tuples and apply min_count threshold
        word_counts = [
            (word, count) for word, count in word_counts.items() if count >= min_count
        ]

        # sort in descending order of word counts
        word_counts = sorted(word_counts, key=lambda wc: -wc[1])
        words = [w[0] for w in word_counts]

        if special_tokens is not None:
            assert len(special_tokens) <= 7, (
                "Can only add less than 7 special tokens. "
                "Change the source code if you want to add more !"
            )
            for i, token in enumerate(special_tokens):
                self.SPECIAL_TOKEN[i] = token

        self.word2index = {}
        self.word2index[self.UNK_TOKEN] = self.UNK_INDEX
        self.word2index[self.PAD_TOKEN] = self.PAD_INDEX
        self.word2index[self.SOS_TOKEN] = self.SOS_INDEX
        self.word2index[self.EOS_TOKEN] = self.EOS_INDEX

        for i in range(len(self.SPECIAL_TOKEN)):
            self.word2index[self.SPECIAL_TOKEN[i]] = self.SPECIAL_TOKEN_INDEX[i]

        for index, word in enumerate(words):
            self.word2index[word] = index + 10
        self.index2word = {index: word for word, index in self.word2index.items()}

    def to_indices(self, words: List[str]) -> List[int]:
        return [self.word2index.get(word.lower(), self.UNK_INDEX) for word in words]

    def to_word(self, indices: List[int]) -> List[str]:
        return [self.index2word.get(index, self.UNK_TOKEN) for index in indices]

    def save(self, save_vocabulary_path: str) -> None:
        # save self.word2index to json file
        with open(save_vocabulary_path, "w") as save_vocabulary_file:
            json.dump(self.word2index, save_vocabulary_file)

    def __len__(self):
        return len(self.index2word)