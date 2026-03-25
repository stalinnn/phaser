#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据训练输出绘制相变监控曲线与双曲嵌入的 2D 投影（PCA 到平面，仅可视化）。

用法:
    cd /phaser/Phase_Transition_Emergence
    python visualize_phase.py --run_dir runs/default_run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_history_json(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_telm(rows: list[dict], out_path: Path) -> None:
    if not rows:
        print("无 TELM 数据，跳过曲线图")
        return
    steps = [r["step"] for r in rows]
    losses = [r["loss"] for r in rows]
    eff = [r["effective_rank"] for r in rows]
    gnorm = [r["grad_norm"] for r in rows]
    mn = [r["hyp_coord_norm_mean"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(steps, losses, "b-", linewidth=1.2)
    axes[0, 0].set_title("Loss (Metric Alignment)")
    axes[0, 0].set_xlabel("step")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, eff, "r-", linewidth=1.2)
    axes[0, 1].set_title("Effective Rank (Order Parameter)")
    axes[0, 1].set_xlabel("step")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, gnorm, "g-", linewidth=1.2)
    axes[1, 0].set_title("Gradient L2 Norm")
    axes[1, 0].set_xlabel("step")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, mn, "m-", linewidth=1.2)
    axes[1, 1].set_title("Hyperbolic Norm Mean (Boundary Monitor)")
    axes[1, 1].set_xlabel("step")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def pca2d(x: np.ndarray) -> np.ndarray:
    """x: [N, d] -> [N, 2]"""
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def plot_embedding_2d(z: np.ndarray, out_path: Path, title: str = "Hyperbolic Embedding PCA-2D") -> None:
    if z.shape[1] < 2:
        print("维数 < 2，跳过 2D 投影图")
        return
    z2 = pca2d(z)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(z2[:, 0], z2[:, 1], s=12, alpha=0.65, c=np.arange(len(z2)), cmap="viridis")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"已保存: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default=str(_EXPERIMENT_ROOT / "runs" / "default_run"),
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir)

    jsonl = run_dir / "telm_readings.jsonl"
    rows = load_jsonl(jsonl)
    if not rows:
        rows = load_history_json(run_dir / "history_summary.json")

    plot_telm(rows, run_dir / "phase_transition_curves.png")

    ckpt = run_dir / "checkpoint.pt"
    if ckpt.is_file():
        try:
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(ckpt, map_location="cpu")
        z = data["z_hyp"].numpy()
        plot_embedding_2d(z, run_dir / "hyperbolic_embedding_pca2d.png")
    else:
        print(f"未找到 {ckpt}，跳过嵌入可视化")


if __name__ == "__main__":
    main()
