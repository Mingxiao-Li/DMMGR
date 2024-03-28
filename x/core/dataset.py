from torch.utils.data import Dataset
from typing import List
import os, json, torch


class XDataset(Dataset):
    def __init__(self, _C):
        super(XDataset, self).__init__()
        self._config = _C

        # set data split
        self._split_subset = ["train", "val", "test"]
        assert _C.split in self._split_subset, "Split can only be train, val or test"
        self._split = _C.split

        if self._split != "test":
            # set dataset path and check if it exists
            data_path = os.path.join(_C.parent_path, _C.path.format(split=self._split))
            self.check_path_exist([data_path])

            # Load data and set number of samples
            self._data = self.load_data_from_json(data_path)

            if _C.num_train_sample != 0 and self._split == "train":
                self._data = self._data[: _C.num_train_sample]
            if _C.num_val_sample != 0 and self._split == "val":
                self._data = self._data[: _C.num_val_sample]

        else:
            test_path = os.path.join(_C.parent_path, _C.path.format(split="test"))
            self.check_path_exist([test_path])
            _test_data = self.load_data_from_json(test_path)
            self._data = _test_data

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, split):
        assert split in self._split_subset, "Split can only be train, val or test."
        self._split = split

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        raise NotImplementedError

    def pad_sequence(
        self, input_list: List, max_seq_len, return_mask: bool = False
    ) :
        input_tensor = torch.LongTensor([input_list])
        if input_tensor.shape[1] >= max_seq_len:
            input_tensor = input_tensor[:, :max_seq_len]
        else:
            input_tensor_zeros = torch.ones(
                1, max_seq_len, dtype=torch.long
            )  # 1 for PAD Token
            input_tensor_zeros[0, : input_tensor.shape[1]] = input_tensor
            input_tensor = input_tensor_zeros

        if return_mask:
            mask = (input_tensor == 1).int()
            return input_tensor, mask
        return input_tensor

    def load_data_from_json(self, data_path):
        r"""A simple method to load json file
        :param data_path:
        :return:
        """
        with open(data_path, "r") as f:
            data = json.load(f)
        return data["data"]

    @staticmethod
    def check_path_exist(paths: List[str]) -> None:
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} deoesn't exist")
