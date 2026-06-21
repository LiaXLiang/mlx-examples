# datasets.py

import json
import os
import random
import urllib.request

import numpy as np


SHIJING_URL = (
    "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master/"
    "%E8%AF%97%E7%BB%8F/shijing.json"
)

BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [
    BOS_TOKEN,
    EOS_TOKEN,
]


class CharVocab:
    def __init__(self, tokens):
        self.itos = list(tokens)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode_chars(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        pieces = []

        for i in ids:
            tok = self.itos[int(i)]

            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue

            pieces.append(tok)

        return "".join(pieces)


def encode_lines_with_special_tokens(lines, vocab):
    encoded = []

    bos_id = vocab.stoi[BOS_TOKEN]
    eos_id = vocab.stoi[EOS_TOKEN]

    for line in lines:
        encoded.append(bos_id)
        encoded.extend(vocab.encode_chars(line))
        encoded.append(eos_id)

    return np.array(encoded, dtype=np.int64)


def load_shijing_dataset(
    data_dir="data",
    valid_ratio=0.1,
    test_ratio=0.1,
    seed=42,
):
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

    random.seed(seed)
    random.shuffle(lines)

    all_text = "".join(lines)
    chars = sorted(set(all_text))

    vocab_tokens = SPECIAL_TOKENS + chars
    vocab = CharVocab(vocab_tokens)

    n = len(lines)
    n_test = int(n * test_ratio)
    n_valid = int(n * valid_ratio)

    test_lines = lines[:n_test]
    valid_lines = lines[n_test : n_test + n_valid]
    train_lines = lines[n_test + n_valid :]

    train = encode_lines_with_special_tokens(train_lines, vocab)
    valid = encode_lines_with_special_tokens(valid_lines, vocab)
    test = encode_lines_with_special_tokens(test_lines, vocab)

    print("Loaded Shijing dataset")
    print(f"Lines: {len(lines)}")
    print(f"Vocab size: {len(vocab)}")
    print(f"Train lines: {len(train_lines)}")
    print(f"Valid lines: {len(valid_lines)}")
    print(f"Test lines: {len(test_lines)}")
    print(f"Train tokens: {len(train)}")
    print(f"Valid tokens: {len(valid)}")
    print(f"Test tokens: {len(test)}")
    print(f"BOS id: {vocab.stoi[BOS_TOKEN]}")
    print(f"EOS id: {vocab.stoi[EOS_TOKEN]}")

    return vocab, train, valid, test


def load_dataset(name):
    if name == "shijing":
        return load_shijing_dataset()

    raise ValueError(f"Unknown dataset: {name}")