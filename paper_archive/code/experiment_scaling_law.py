"""
EXP: TGN Scaling Law on TinyShakespeare (paper section 6.3)
----------------------------------------------------------
This file was rewritten to be:
- runnable as a standalone script
- reproducible (seeded + train/val split)
- explicit about what "gate usage" means
- safe (does NOT overwrite manuscript figures by default)

It produces:
- CSV: paper_archive/results/scaling_law_results.csv
- Figure: paper_archive/figures/inverse_scaling_law_hyper_sparsity.png

NOTE: This is still a lightweight experiment (fixed-step budget), not a full SOTA LM training pipeline.
"""

from __future__ import annotations
import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_project_root() -> Path:
    """
    Resolve project root robustly across:
    - normal script execution (has __file__)
    - notebooks / interactive shells (no __file__)

    We define "project root" as the nearest parent directory that contains `paper_archive/`.
    """
    try:
        start = Path(__file__).resolve().parent
    except NameError:
        # Jupyter / interactive
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
        self.text = text
        self.block_size = block_size
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self) -> int:
        # number of possible blocks
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
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.split(C, dim=-1)

        # (B, nh, T, hd)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v  # (B, nh, T, hd)
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


class TGNBlock(nn.Module):
    """
    A minimal "TGN-like" Transformer block with a learnable gate per token:
      y = x + g ⊙ Attn(LN(x))
      z = y + MLP(LN(y))

    Gate is produced by a tiny MLP on LN(x) and trained with an energy penalty.
    This matches the paper's narrative ("gate usage" is mean gate activation).
    """

    def __init__(self, n_embd: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.gate = nn.Sequential(
            nn.Linear(n_embd, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.ln1(x)
        g = self.gate(h)  # (B,T,1)
        x = x + g * self.attn(h)
        x = x + self.mlp(self.ln2(x))
        return x, g


class TGNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, n_head: int, n_layer: int, block_size: int, dropout: float):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TGNBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.block_size
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))

        gates: List[torch.Tensor] = []
        for block in self.blocks:
            x, g = block(x)
            gates.append(g)

        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, gates


@torch.no_grad()
def estimate_val_loss_and_gate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    gate_threshold: float,
    batches: int = 10,
) -> Tuple[float, float, float]:
    model.eval()
    losses = []
    gates_mean = []
    gates_rate = []
    it = iter(loader)
    for _ in range(batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            break
        xb, yb = xb.to(device), yb.to(device)
        _, loss, g_list = model(xb, yb)
        losses.append(loss.item())
        g_stack = torch.stack([g.mean() for g in g_list]).mean()
        g_rate = torch.stack([(g > gate_threshold).float().mean() for g in g_list]).mean()
        gates_mean.append(float(g_stack.item()))
        gates_rate.append(float(g_rate.item()))
    model.train()
    return float(np.mean(losses)), float(np.mean(gates_mean)), float(np.mean(gates_rate))


def run_scaling_experiment():
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
    ENERGY_PENALTY = 5e-3  # λ · mean(gates)
    GATE_THRESHOLD = 0.5
    TARGET_EFFECTIVE_BATCH = 64  # via grad accumulation
    SKIP_LARGE = os.environ.get("SKIP_LARGE", "0").strip() == "1"

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    total_vram_gb = None
    if device.type == "cuda":
        total_vram_gb = float(torch.cuda.get_device_properties(0).total_memory / 1e9)
        print(f"GPU Memory: {total_vram_gb:.2f} GB")

    # 1) Prepare Data (TinyShakespeare)
    out_root = get_project_root()
    data_path = out_root / "data" / "tinyshakespeare_input.txt"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading dataset to {data_path} ...")
        r = requests.get(data_url, timeout=30)
        r.raise_for_status()
        data_path.write_text(r.text, encoding="utf-8")

    text = data_path.read_text(encoding="utf-8")

    # 2) Configurations (match paper naming; actual params are measured and reported)
    configs = [
        {"name": "Tiny (1M)", "n_layer": 4, "n_head": 4, "n_embd": 128, "batch_size": 64},
        {"name": "Small (21M)", "n_layer": 8, "n_head": 8, "n_embd": 384, "batch_size": 32},
        {"name": "Medium (128M)", "n_layer": 12, "n_head": 12, "n_embd": 768, "batch_size": 16},
        {"name": "Large (454M)", "n_layer": 24, "n_head": 16, "n_embd": 1024, "batch_size": 4},
    ]
    if SKIP_LARGE:
        configs = [c for c in configs if not c["name"].startswith("Large")]
        print("[Config] SKIP_LARGE=1, skipping Large config.")

    # Safety: skip the largest config on smaller GPUs to avoid OOM during local reproduction.
    # You can always re-enable it on a bigger machine by setting MIN_VRAM_GB_FOR_LARGE lower.
    MIN_VRAM_GB_FOR_LARGE = 20.0
    if device.type == "cuda" and total_vram_gb is not None and total_vram_gb < MIN_VRAM_GB_FOR_LARGE:
        configs = [c for c in configs if not c["name"].startswith("Large")]
        print(f"[Safety] VRAM<{MIN_VRAM_GB_FOR_LARGE:.0f}GB, skipping Large config to avoid OOM.")

    dataset = CharDataset(text, BLOCK_SIZE)
    n_val = int(len(dataset) * VAL_FRAC)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    results = []  # rows for CSV

    for conf_idx, conf in enumerate(configs):
        print(f"\n--- Training {conf['name']} ---")
        train_loader = DataLoader(train_set, batch_size=conf["batch_size"], shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=conf["batch_size"], shuffle=False, num_workers=0, drop_last=True)

        model = TGNLanguageModel(
            vocab_size=dataset.vocab_size,
            n_embd=conf["n_embd"],
            n_head=conf["n_head"],
            n_layer=conf["n_layer"],
            block_size=BLOCK_SIZE,
            dropout=DROPOUT,
        ).to(device)

        param_count_m = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Actual Parameters: {param_count_m:.2f}M")

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        model.train()

        it = iter(train_loader)
        train_losses_tail = []
        gate_tail = []
        gate_rate_tail = []
        val_losses = []
        val_gates = []
        val_gate_rates = []

        t0 = time.perf_counter()
        max_iters = MAX_ITERS_LARGE if conf["name"].startswith("Large") else MAX_ITERS
        accum_steps = max(1, int(math.ceil(TARGET_EFFECTIVE_BATCH / conf["batch_size"])))
        print(
            f"Max iters: {max_iters} | batch={conf['batch_size']} | grad_accum={accum_steps} "
            f"(effective batch≈{conf['batch_size'] * accum_steps})"
        )

        for step in range(max_iters):
            optimizer.zero_grad(set_to_none=True)

            # accumulate statistics across micro-steps
            loss_item = 0.0
            gate_mean_item = 0.0
            gate_rate_item = 0.0

            for _ in range(accum_steps):
                try:
                    xb, yb = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    xb, yb = next(it)

                xb, yb = xb.to(device), yb.to(device)
                _, loss, gates = model(xb, yb)

                g_stack = torch.stack([g.mean() for g in gates]).mean()
                g_rate = torch.stack([(g > GATE_THRESHOLD).float().mean() for g in gates]).mean()

                loss_total = loss + ENERGY_PENALTY * g_stack
                (loss_total / accum_steps).backward()

                loss_item += float(loss.item())
                gate_mean_item += float(g_stack.item())
                gate_rate_item += float(g_rate.item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_item /= accum_steps
            gate_mean_item /= accum_steps
            gate_rate_item /= accum_steps

            if step % EVAL_INTERVAL == 0 or step == max_iters - 1:
                val_loss, val_gate_mean, val_gate_rate = estimate_val_loss_and_gate(
                    model, val_loader, device, gate_threshold=GATE_THRESHOLD, batches=EVAL_BATCHES
                )
                val_losses.append(val_loss)
                val_gates.append(val_gate_mean)
                val_gate_rates.append(val_gate_rate)
                print(
                    f"Iter {step:4d}: train_loss={loss_item:.4f}, val_loss={val_loss:.4f}, "
                    f"gate_mean={gate_mean_item:.4f}, gate_rate={gate_rate_item:.4f}, "
                    f"val_gate_mean={val_gate_mean:.4f}, val_gate_rate={val_gate_rate:.4f}"
                )

            if step >= max_iters - 50:
                train_losses_tail.append(loss_item)
                gate_tail.append(gate_mean_item)
                gate_rate_tail.append(gate_rate_item)

        dt = time.perf_counter() - t0
        final_train_loss = float(np.mean(train_losses_tail)) if train_losses_tail else float("nan")
        final_gate = float(np.mean(gate_tail)) if gate_tail else float("nan")
        final_gate_rate = float(np.mean(gate_rate_tail)) if gate_rate_tail else float("nan")
        final_val_loss = float(val_losses[-1]) if val_losses else float("nan")
        final_val_gate = float(val_gates[-1]) if val_gates else float("nan")
        final_val_gate_rate = float(val_gate_rates[-1]) if val_gate_rates else float("nan")

        results.append(
            {
                "name": conf["name"],
                "params_m": param_count_m,
                "final_train_loss": final_train_loss,
                "final_val_loss": final_val_loss,
                "final_gate_mean": final_gate,
                "final_gate_rate": final_gate_rate,
                "final_val_gate_mean": final_val_gate,
                "final_val_gate_rate": final_val_gate_rate,
                "seconds": dt,
                "max_iters": max_iters,
                "block_size": BLOCK_SIZE,
                "seed": SEED,
                "energy_penalty": ENERGY_PENALTY,
                "gate_threshold": GATE_THRESHOLD,
                "grad_accum_steps": accum_steps,
            }
        )

        # Cleanup GPU memory except keep last model for optional probing
        if conf_idx < len(configs) - 1:
            del model, optimizer, xb, yb, loss, gates
            if device.type == "cuda":
                torch.cuda.empty_cache()
        else:
            last_model = model
            last_val_loader = val_loader

    # 3) Print + Save CSV
    print("\n--- Scaling Law Results (validation) ---")
    print(f"{'Model':<16} | {'Params(M)':<9} | {'ValLoss':<10} | {'GateMean':<8} | {'GateRate':<8}")
    print("-" * 70)
    for row in results:
        print(
            f"{row['name']:<16} | {row['params_m']:<9.2f} | {row['final_val_loss']:<10.4f} | "
            f"{row['final_val_gate_mean']:<8.4f} | {row['final_val_gate_rate']:<8.4f}"
        )

    results_dir = out_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "scaling_law_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved CSV to {csv_path}")

    # 4) Plot inverse scaling (val loss vs params) + gate usage
    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "inverse_scaling_law_hyper_sparsity.png"

    params = np.array([r["params_m"] for r in results])
    val_loss = np.array([r["final_val_loss"] for r in results])
    # Use hard activation rate as the primary "sparsity" metric (paper claim)
    gate = np.array([r["final_gate_rate"] for r in results]) * 100.0  # %

    plt.style.use("seaborn-v0_8-paper")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(params, val_loss, "o-", color="#1f77b4", label="Validation Loss")
    ax2.plot(params, gate, "s--", color="#d62728", label=f"Gate Activation Rate (%) (g>{results[0]['gate_threshold']})")

    ax1.set_xscale("log")
    ax1.set_xlabel("Actual Parameters (Millions)")
    ax1.set_ylabel("Validation Loss")
    ax2.set_ylabel("Gate Activation Rate (%)")
    ax1.set_title("Inverse Scaling Law: Hyper-Sparsity Emergence", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper center", ncol=2, frameon=False)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    print(f"Saved figure to {fig_path}")

    # 5) Optional: probe layer-wise gate on the largest model (one batch)
    if "last_model" in locals():
        thr = float(results[-1].get("gate_threshold", 0.5))
        print(f"\n--- Deep Probing of {results[-1]['name']} (Layer-wise Gate: mean(g)% and rate(g>{thr})% ) ---")
        last_model.eval()
        xb, yb = next(iter(last_val_loader))
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            _, _, gates = last_model(xb, yb)
        gate_means = [float(g.mean().item() * 100) for g in gates]
        gate_rates = [float((g > thr).float().mean().item() * 100) for g in gates]
        for li, (m, r) in enumerate(zip(gate_means, gate_rates)):
            print(f"Layer {li:>2d}: mean={m:.4f}% | rate={r:.4f}%")
        if len(gate_means) >= 8:
            bottom_m = float(np.mean(gate_means[:4]))
            top_m = float(np.mean(gate_means[-4:]))
            bottom_r = float(np.mean(gate_rates[:4]))
            top_r = float(np.mean(gate_rates[-4:]))
            print(f"Bottom Layers Avg: mean={bottom_m:.4f}% | rate={bottom_r:.4f}%")
            print(f"Top Layers Avg:    mean={top_m:.4f}% | rate={top_r:.4f}%")


if __name__ == "__main__":
    run_scaling_experiment()
