# datasets.py

import json
import os
import urllib.request

import numpy as np


SHIJING_URL = (
    "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/"
    "%E8%AF%97%E7%BB%8F/shijing.json"
)

SPECIAL_TOKENS = [
    "<BOS>",
    "<EOS>",
]


class CharVocab:
    def __init__(self, chars):
        self.itos = list(chars)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


def load_shijing_dataset(data_dir="data", valid_ratio=0.1, test_ratio=0.1):
    os.makedirs(data_dir, exist_ok=True)

    json_path = os.path.join(data_dir, "shijing.json")

    if not os.path.exists(json_path):
        print("Downloading shijing.json...")
        urllib.request.urlretrieve(SHIJING_URL, json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = [
        line.strip()
        for item in data
        for line in item["content"]
        if line.strip()
    ]

    text = "\n".join(lines)

    chars = sorted(set(text))
    vocab = CharVocab(chars)

    encoded = np.array(vocab.encode(text), dtype=np.int64)

    n = len(encoded)
    n_test = int(n * test_ratio)
    n_valid = int(n * valid_ratio)
    n_train = n - n_valid - n_test

    train = encoded[:n_train]
    valid = encoded[n_train : n_train + n_valid]
    test = encoded[n_train + n_valid :]

    print(f"Loaded Shijing dataset")
    print(f"Lines: {len(lines)}")
    print(f"Characters: {len(text)}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Train tokens: {len(train)}")
    print(f"Valid tokens: {len(valid)}")
    print(f"Test tokens: {len(test)}")

    return vocab, train, valid, test


def load_dataset(name):
    if name == "shijing":
        return load_shijing_dataset()

    raise ValueError(f"Unknown dataset: {name}")