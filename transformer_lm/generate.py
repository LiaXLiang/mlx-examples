# generate.py

import argparse
import torch
import torch.nn.functional as F

from train import TransformerLM


BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [BOS_TOKEN, EOS_TOKEN]


class CharVocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}

    def __len__(self):
        return len(self.itos)

    def encode_chars(self, text):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids, skip_special_tokens=True):
        pieces = []

        for i in ids:
            tok = self.itos[int(i)]

            if skip_special_tokens and tok in SPECIAL_TOKENS:
                continue

            pieces.append(tok)

        return "".join(pieces)


@torch.no_grad()
def generate(
    model,
    vocab,
    prompt,
    max_new_tokens,
    context_size,
    device,
):
    model.eval()

    bos_id = vocab.stoi[BOS_TOKEN]
    eos_id = vocab.stoi[EOS_TOKEN]

    ids = [bos_id]
    ids.extend(vocab.encode_chars(prompt))

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        x_cond = x[:, -context_size:]

        logits = model(x_cond)
        logits = logits[:, -1, :]

        next_id = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = next_id.item()
        x = torch.cat([x, next_id], dim=1)

        if token_id == eos_id:
            break

    return vocab.decode(x[0].tolist(), skip_special_tokens=True)


def load_model(args):
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(args.checkpoint, map_location=device)

    train_args = checkpoint["args"]
    vocab = CharVocab(checkpoint["vocab_itos"])

    model = TransformerLM(
        vocab_size=len(vocab),
        num_layers=train_args["num_blocks"],
        dims=train_args["dim"],
        num_heads=train_args["num_heads"],
        context_size=train_args["context_size"],
        dropout=train_args.get("dropout", 0.0),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab, train_args, device


def interactive_loop(model, vocab, train_args, device, args):
    print("Interactive Shijing LM")
    print("输入 prompt 后回车生成。输入 :q 退出。")
    print("输入 :settings 查看当前参数。")
    print("输入 :temp 0.8 修改 temperature。")
    print("输入 :topk 20 修改 top_k。")
    print("输入 :len 100 修改生成长度。")
    print("-" * 50)

    # temperature = args.temperature
    # top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    context_size = train_args["context_size"]

    while True:
        prompt = input("\nPrompt > ").strip()

        if prompt in {":q", ":quit", "exit", "quit"}:
            print("Bye.")
            break

        if prompt == ":settings":
            print(f"temperature = {temperature}")
            print(f"top_k = {top_k}")
            print(f"max_new_tokens = {max_new_tokens}")
            print(f"context_size = {context_size}")
            print(f"device = {device}")
            continue

        if prompt.startswith(":temp "):
            temperature = float(prompt.split(maxsplit=1)[1])
            print(f"temperature updated to {temperature}")
            continue

        if prompt.startswith(":topk "):
            top_k = int(prompt.split(maxsplit=1)[1])
            print(f"top_k updated to {top_k}")
            continue

        if prompt.startswith(":len "):
            max_new_tokens = int(prompt.split(maxsplit=1)[1])
            print(f"max_new_tokens updated to {max_new_tokens}")
            continue

        if not prompt:
            print("请输入一些起始文字，比如：关关 / 蒹葭 / 桃之夭夭")
            continue

        output = generate(
            model=model,
            vocab=vocab,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            device=device,
        )

        print("\nGenerated:")
        print(output)


def main(args):
    model, vocab, train_args, device = load_model(args)
    interactive_loop(model, vocab, train_args, device, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Interactive Shijing text generation.")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/shijing_lm.pt",
    )

    parser.add_argument("--max_new_tokens", type=int, default=100)
    # parser.add_argument("--temperature", type=float, default=1.0)
    # parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()
    main(args)