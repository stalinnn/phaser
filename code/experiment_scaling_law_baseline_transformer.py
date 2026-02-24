"""
Baseline for paper 6.3 (NO TGN / NO gate):
-----------------------------------------
Standard causal Transformer language model trained on TinyShakespeare with a fixed-step budget.

This script is designed to be:
  - standalone + notebook-friendly (no reliance on __file__ in notebooks)
  - reproducible (seeded + train/val split)
  - comparable to the TGN scripts in this repo (same dataset, same training budget, same configs)

Outputs:
  - CSV: paper_archive/results/scaling_law_results_baseline_transformer.csv
  - Figure: paper_archive/figures/inverse_scaling_baseline_transformer.png
"""

from __future__ import annotations

import csv
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt


# =========================
# Hard-coded paper settings
# =========================
SEED = 42
MAX_ITERS = 1000
MAX_ITERS_LARGE = 5000
BLOCK_SIZE = 256
LR = 3e-4
DROPOUT = 0.1
VAL_FRAC = 0.1
EVAL_INTERVAL = 100
EVAL_BATCHES = 10
GRAD_CLIP = 1.0
TARGET_EFFECTIVE_BATCH = 64  # via grad accumulation

# Optional: set environment variable SKIP_LARGE=1 to skip the slowest Large config.
SKIP_LARGE = os.environ.get("SKIP_LARGE", "0").strip() == "1"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_project_root() -> Path:
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        start = Path.cwd()

    if (start / "paper_archive").exists():
        return start
    for p in [start] + list(start.parents):
        if (p / "paper_archive").exists():
            return p
    return start


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int):
        super().__init__()
        self.block_size = block_size
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self) -> int:
        return max(0, self.data.numel() - self.block_size - 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,nh,T,T)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.dropout(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int, dropout: float):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok(idx) + self.pos(pos))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


@torch.no_grad()
def estimate_val(model: nn.Module, loader: DataLoader, device: torch.device, batches: int = 10) -> float:
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            break
        xb, yb = xb.to(device), yb.to(device)
        _, loss = model(xb, yb)
        losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def run():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    total_vram_gb = None
    if device.type == "cuda":
        total_vram_gb = float(torch.cuda.get_device_properties(0).total_memory / 1e9)
        print(f"GPU Memory: {total_vram_gb:.2f} GB")

    root = get_project_root()
    data_path = root / "data" / "tinyshakespeare_input.txt"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading dataset to {data_path} ...")
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data_path.write_text(r.text, encoding="utf-8")

    text = data_path.read_text(encoding="utf-8")
    dataset = CharDataset(text, BLOCK_SIZE)
    n_val = int(len(dataset) * VAL_FRAC)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    configs = [
        {"name": "Tiny (1M)", "n_layer": 4, "n_head": 4, "n_embd": 128, "batch_size": 64},
        {"name": "Small (21M)", "n_layer": 8, "n_head": 8, "n_embd": 384, "batch_size": 32},
        {"name": "Medium (128M)", "n_layer": 12, "n_head": 12, "n_embd": 768, "batch_size": 16},
        {"name": "Large (454M)", "n_layer": 24, "n_head": 16, "n_embd": 1024, "batch_size": 4},
    ]
    if SKIP_LARGE:
        configs = [c for c in configs if not c["name"].startswith("Large")]
        print("[Config] SKIP_LARGE=1, skipping Large config.")
    # Safety: skip largest config on smaller GPUs to avoid OOM in local reproduction.
    MIN_VRAM_GB_FOR_LARGE = 20.0
    if device.type == "cuda" and total_vram_gb is not None and total_vram_gb < MIN_VRAM_GB_FOR_LARGE:
        configs = [c for c in configs if not c["name"].startswith("Large")]
        print(f"[Safety] VRAM<{MIN_VRAM_GB_FOR_LARGE:.0f}GB, skipping Large config to avoid OOM.")

    results: List[Dict[str, float | str]] = []

    for conf_idx, conf in enumerate(configs):
        print(f"\n--- Training {conf['name']} ---")
        train_loader = DataLoader(train_set, batch_size=conf["batch_size"], shuffle=True, drop_last=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=conf["batch_size"], shuffle=False, drop_last=True, num_workers=0)

        model = TransformerLM(
            vocab_size=dataset.vocab_size,
            n_embd=conf["n_embd"],
            n_head=conf["n_head"],
            n_layer=conf["n_layer"],
            block_size=BLOCK_SIZE,
            dropout=DROPOUT,
        ).to(device)

        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Actual Parameters: {params_m:.2f}M")

        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        it = iter(train_loader)
        tail_loss = []
        val_last = float("nan")

        t0 = time.perf_counter()
        max_iters = MAX_ITERS_LARGE if conf["name"].startswith("Large") else MAX_ITERS
        accum_steps = max(1, int(math.ceil(TARGET_EFFECTIVE_BATCH / conf["batch_size"])))
        print(
            f"Max iters: {max_iters} | batch={conf['batch_size']} | grad_accum={accum_steps} "
            f"(effective batch≈{conf['batch_size'] * accum_steps})"
        )

        for step in range(max_iters):
            opt.zero_grad(set_to_none=True)
            loss_item = 0.0

            for _ in range(accum_steps):
                try:
                    xb, yb = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    xb, yb = next(it)

                xb, yb = xb.to(device), yb.to(device)
                _, loss = model(xb, yb)

                (loss / accum_steps).backward()
                loss_item += float(loss.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            loss_item /= accum_steps

            if step % EVAL_INTERVAL == 0 or step == max_iters - 1:
                val_last = estimate_val(model, val_loader, device, batches=EVAL_BATCHES)
                print(f"Iter {step:4d}: train_loss={loss_item:.4f}, val_loss={val_last:.4f}")

            if step >= max_iters - 50:
                tail_loss.append(float(loss_item))

        seconds = time.perf_counter() - t0

        results.append(
            {
                "name": conf["name"],
                "params_m": float(params_m),
                "final_train_loss": float(np.mean(tail_loss)) if tail_loss else float("nan"),
                "final_val_loss": float(val_last),
                "max_iters": float(max_iters),
                "block_size": float(BLOCK_SIZE),
                "seed": float(SEED),
                "grad_accum_steps": float(accum_steps),
                "seconds": float(seconds),
            }
        )

        if conf_idx < len(configs) - 1:
            del model, opt, xb, yb, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Save CSV
    out_root = get_project_root()
    results_dir = out_root / "paper_archive" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "scaling_law_results_baseline_transformer.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved CSV to {csv_path}")

    # Plot
    fig_dir = out_root / "paper_archive" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "inverse_scaling_baseline_transformer.png"

    params = np.array([r["params_m"] for r in results], dtype=float)
    vloss = np.array([r["final_val_loss"] for r in results], dtype=float)

    plt.style.use("seaborn-v0_8-paper")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(params, vloss, "o-", color="#1f77b4", label="Validation Loss")
    ax.set_xscale("log")
    ax.set_xlabel("Actual Parameters (Millions)")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Baseline Transformer: Scaling (Validation Loss vs Params)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    run()

