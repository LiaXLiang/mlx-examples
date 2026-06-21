import math
import time
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dataset


# 没有tailored to new dataset之前是标准的casual LM: 
# 输入前 n 个 token，预测下一个 token；模型只关心 vocab_size 和 token 序列，不关心数据来自诗歌还是 quote。
# 它通过 dataset.load_dataset(args.dataset) 加载数据，之后统一切成 context_size+1 的窗口训练。




class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 10000):
        super().__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).float().unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, L, D]
        L = x.size(1)
        return self.pe[:L].unsqueeze(0)


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        context_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = SinusoidalPositionalEncoding(dims, max_len=context_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dims,
            nhead=num_heads,
            dim_feedforward=4 * dims,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out_proj = nn.Linear(dims, vocab_size)

    def forward(self, x):
        # x: [B, L]
        B, L = x.shape

        # causal mask: [L, L]
        causal_mask = torch.triu(
            torch.ones(L, L, device=x.device, dtype=torch.bool),
            diagonal=1,
        )

        x = self.embedding(x)
        x = x + self.pe(x)
        x = self.transformer(x, mask=causal_mask)
        return self.out_proj(x)


def to_samples(context_size, dataset):
    window_size = context_size + 1
    samples = dataset.size // window_size
    dataset = dataset[: samples * window_size]
    dataset = dataset.reshape(samples, -1)

    return torch.from_numpy(dataset.astype("int64"))


def iterate_batches(batch_size, context_size, dataset, device):
    inputs = to_samples(context_size, dataset)

    s = 0
    while True:
        if s == 0:
            perm = torch.randperm(inputs.shape[0])

        ids = perm[s : s + batch_size]
        yield inputs[ids].to(device)

        s += batch_size
        if s >= inputs.shape[0]:
            s = 0


def main(args):
    torch.manual_seed(args.seed)

    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.gpu and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    batch_size = args.batch_size
    context_size = args.context_size

    vocab, train, valid, test = dataset.load_dataset(args.dataset)

    model = TransformerLM(
        vocab_size=len(vocab),
        num_layers=args.num_blocks,
        dims=args.dim,
        num_heads=args.num_heads,
        context_size=context_size,
        dropout=args.dropout,
    ).to(device)

    nparams = sum(
        p.numel()
        for name, p in model.named_parameters()
        if "embedding" not in name
    )

    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    def loss_fn(inputs, reduction="mean"):
        x = inputs[..., :-1]
        y = inputs[..., 1:]

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction=reduction,
        )

        return loss

    @torch.no_grad()
    def eval_fn(dataset):
        model.eval()

        inputs = to_samples(context_size, dataset).to(device)

        total_loss = 0.0
        total_tokens = 0

        for s in range(0, inputs.shape[0], batch_size):
            batch = inputs[s : s + batch_size]
            loss = loss_fn(batch, reduction="sum")

            total_loss += loss.item()
            total_tokens += batch[:, 1:].numel()

        model.train()
        return total_loss / total_tokens

    def train_step(inputs):
        optimizer.zero_grad(set_to_none=True)

        loss = loss_fn(inputs)
        loss.backward()

        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        return loss

    train_iterator = iterate_batches(
        batch_size=batch_size,
        context_size=context_size,
        dataset=train,
        device=device,
    )

    model.train()

    losses = []
    tic = time.perf_counter()

    for it, inputs in zip(range(args.num_iters), train_iterator):
        lr = min(1.0, it / args.lr_warmup) * args.learning_rate

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        loss = train_step(inputs)
        losses.append(loss.item())

        if (it + 1) % args.steps_per_report == 0:
            train_loss = sum(losses) / len(losses)
            toc = time.perf_counter()

            print(
                f"Iter {it + 1}: "
                f"Train loss {train_loss:.3f}, "
                f"It/sec {args.steps_per_report / (toc - tic):.3f}"
            )

            losses = []
            tic = time.perf_counter()

        if (it + 1) % args.steps_per_eval == 0:
            val_loss = eval_fn(valid)
            toc = time.perf_counter()

            print(
                f"Iter {it + 1}: "
                f"Val loss {val_loss:.3f}, "
                f"Val ppl {math.exp(val_loss):.3f}, "
                f"Val took {(toc - tic):.3f}s"
            )

            tic = time.perf_counter()

    if args.eval_test:
        test_loss = eval_fn(test)
        test_ppl = math.exp(test_loss)

        print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")

    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_itos": vocab.itos,
        "args": vars(args),
    }
    save_path = os.path.join(args.save_dir, "shijing_lm.pt")
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a decoder-only Transformer LM with PyTorch.")

    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--dataset",
        type=str,
        default="shijing",
    )

    parser.add_argument("--context_size", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=2)

    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_iters", type=int, default=600)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_warmup", type=int, default=10)

    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=None)

    parser.add_argument("--steps_per_report", type=int, default=5)
    parser.add_argument("--steps_per_eval", type=int, default=20)

    parser.add_argument("--eval_test", action="store_true")

    parser.add_argument("--save_dir", type=str, default="checkpoints")

    args = parser.parse_args()
    main(args)
