from abc import ABC
import os
import string

from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ViBaDataset(Dataset, ABC):
    def __init__(self, data_folder, mode, tokenizer_path: str):
        super(ViBaDataset, self).__init__()
        self.data = self.load_data(data_folder, mode)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    @staticmethod
    def load_data(data_folder, mode):
        vi_data_file = [item for item in os.listdir(data_folder) if mode in item and ".vi" in item][0]
        ba_data_file = [item for item in os.listdir(data_folder) if mode in item and ".ba" in item][0]
        vi_data_path = os.path.join(data_folder, vi_data_file)
        ba_data_path = os.path.join(data_folder, ba_data_file)
        vi_data = [item.replace("\n", "").strip() for item in open(vi_data_path, "r", encoding="utf8").readlines()]
        ba_data = [item.replace("\n", "").strip() for item in open(ba_data_path, "r", encoding="utf8").readlines()]
        return list(zip(vi_data, ba_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vi, ba = self.data[idx]
        for p in string.punctuation:
            vi = vi.replace(p, f" {p} ")
            ba = ba.replace(p, f" {p} ")
        while "  " in vi:
            vi = vi.replace("  ", " ")
        while "  " in ba:
            ba = ba.replace("  ", " ")
        vi = vi.strip()
        ba = ba.strip()
        vi_tokenize = self.tokenizer(vi)
        ba_tokenize = self.tokenizer(ba)
        return {
            "input_ids": vi_tokenize.input_ids,
            "attention_mask": vi_tokenize.attention_mask,
            "labels": ba_tokenize.input_ids
        }

    @staticmethod
    def get_datasets(data_folder, tokenizer_path):
        train_dataset = ViBaDataset(data_folder, "train", tokenizer_path)
        valid_dataset = ViBaDataset(data_folder, "valid", tokenizer_path)
        test_dataset = ViBaDataset(data_folder, "test", tokenizer_path)
        return train_dataset, valid_dataset, test_dataset


if __name__ == "__main__":
    dataset = ViBaDataset(data_folder="../data", mode="train", tokenizer_path="../pretrained/bartpho_syllable")
    print(dataset[0])
