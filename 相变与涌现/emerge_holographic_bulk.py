#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全息相变涌现训练主脚本：边界欧氏语义相似度 -> 体空间双曲度规对齐 (RT 映射离散版)。

运行方式（请在实验目录下执行，以便 import core / probes）:
    cd /phaser/Phase_Transition_Emergence && python emerge_holographic_bulk.py

可选：--data_path 指向每行一段文本的 .txt 或 .jsonl（含 "text" 字段）。
若无数据或 BGE 不可用，自动使用合成欧氏嵌入以保证可复现演示。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 保证从任意工作目录运行时可找到 core / probes
_EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

import geoopt

from core.hyperbolic_space import clamp_to_ball, make_poincare_ball
from core.metric_alignment import (
    cosine_similarity_matrix,
    map_similarity_to_entanglement_proxy,
    metric_alignment_loss,
    upper_triangle_mask,
)
from probes.telm_monitor import collect_reading


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_texts(data_path: str | None, max_nodes: int) -> list[str]:
    """加载短文本列表；若无文件则返回合成模板句。"""
    texts: list[str] = []
    if data_path and os.path.isfile(data_path):
        path = Path(data_path)
        if path.suffix.lower() == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if len(texts) >= max_nodes:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        t = obj.get("text", obj.get("content", ""))
                    except json.JSONDecodeError:
                        t = line
                    if t:
                        texts.append(str(t)[:512])
        else:
            with open(path, "r", encoding="utf-8") as f:
                buf = f.read().split("\n\n")
                for chunk in buf:
                    chunk = chunk.strip().replace("\n", " ")
                    if chunk:
                        texts.append(chunk[:512])
                    if len(texts) >= max_nodes:
                        break
    while len(texts) < max_nodes:
        # 合成层级/主题弱结构：重复词族使部分 pair 余弦偏高
        tid = len(texts)
        texts.append(
            f"实体A_{tid % 7} 与 关系R_{tid % 5} 导致 结果C_{tid % 11}。"
            f" 补充细节编号 {tid}。"
        )
    return texts[:max_nodes]


def encode_euclidean(
    texts: list[str],
    device: torch.device,
    offline: bool = True,
    force_synthetic: bool = False,
) -> torch.Tensor:
    """BGE-M3 编码；失败或 force_synthetic 则返回结构化合成欧氏嵌入。"""
    if force_synthetic:
        print("[INFO] 使用合成欧氏嵌入 (--synthetic_euclidean)")
        return _synthetic_euclidean_embeddings(len(texts), device)
    try:
        from sentence_transformers import SentenceTransformer

        os.environ.setdefault("HF_HUB_OFFLINE", "1" if offline else "0")
        model = SentenceTransformer(
            "BAAI/bge-m3",
            device=str(device),
            local_files_only=offline,
        )
        model.max_seq_length = 256
        embs = model.encode(
            texts,
            batch_size=16,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=False,
        )
        if isinstance(embs, torch.Tensor):
            return embs.float().to(device)
        return torch.tensor(np.asarray(embs), dtype=torch.float32, device=device)
    except Exception as e:
        print(f"[WARN] BGE-M3 不可用，使用合成欧氏嵌入: {e}")
        return _synthetic_euclidean_embeddings(len(texts), device)


def _synthetic_euclidean_embeddings(n: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(42)
    d = 128
    raw = torch.randn(n, d, device=device, generator=g)
    cluster = torch.arange(n, device=device) % 8
    centers = torch.randn(8, d, device=device, generator=g) * 0.5
    return centers[cluster] + raw * 0.3


def build_target_proxy(euc_emb: torch.Tensor, shift_positive: bool = True) -> torch.Tensor:
    M = cosine_similarity_matrix(euc_emb)
    return map_similarity_to_entanglement_proxy(M, shift_positive=shift_positive)


def train(
    num_nodes: int,
    hyp_dim: int,
    epochs: int,
    lr: float,
    batch_size: int,
    telm_every: int,
    data_path: str | None,
    run_dir: Path,
    device: torch.device,
    c: float = 1.0,
    use_upper_triangle_only: bool = True,
    seed: int = 42,
    force_synthetic: bool = False,
) -> None:
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    texts = load_texts(data_path, num_nodes)
    print(f"节点数 N={len(texts)}, 双曲维数 d={hyp_dim}")

    euc = encode_euclidean(texts, device, force_synthetic=force_synthetic)
    target = build_target_proxy(euc)
    n = target.size(0)
    mask = (
        upper_triangle_mask(n, device=device)
        if use_upper_triangle_only
        else None
    )

    manifold = make_poincare_ball(c=c)
    # 小范数初始化，避免一出球
    init = torch.randn(n, hyp_dim, device=device) * 0.02
    init = clamp_to_ball(init)
    z = geoopt.ManifoldParameter(init, manifold=manifold)
    optimizer = geoopt.optim.RiemannianAdam([z], lr=lr)

    history: list[dict] = []
    jsonl_path = run_dir / "telm_readings.jsonl"

    print("开始度规对齐训练 (exp(-d_hyp) ≈ 边界纠缠代理)...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = metric_alignment_loss(manifold, z, target, mask=mask)
        loss.backward()
        optimizer.step()
        # 投影回球内（数值安全）
        with torch.no_grad():
            z.data.copy_(clamp_to_ball(z.data))

        step = epoch + 1
        if step % telm_every == 0 or step == 1:
            reading = collect_reading(
                step=step,
                loss=loss.item(),
                z_hyp=z.data,
                params_with_grad=[z],
            )
            rec = reading.to_json_dict()
            rec["epoch"] = epoch + 1
            history.append(rec)
            with open(jsonl_path, "a", encoding="utf-8") as fj:
                fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(
                f"epoch {step:4d}  loss={loss.item():.6f}  "
                f"eff_rank={reading.effective_rank:.4f}  "
                f"grad_norm={reading.grad_norm:.6f}  "
                f"||x||_mean={reading.hyp_coord_norm_mean:.4f}"
            )

    # 保存双曲坐标与目标矩阵供可视化
    torch.save(
        {
            "z_hyp": z.detach().cpu(),
            "target_proxy": target.cpu(),
            "euc_emb": euc.detach().cpu(),
            "texts": texts,
            "config": {
                "num_nodes": n,
                "hyp_dim": hyp_dim,
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "c": c,
            },
        },
        run_dir / "checkpoint.pt",
    )
    with open(run_dir / "history_summary.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    print(f"完成。日志: {jsonl_path}, 权重: {run_dir / 'checkpoint.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="全息相变涌现训练 (度规对齐)")
    parser.add_argument("--num_nodes", type=int, default=128, help="节点数 N（全矩阵 O(N^2)）")
    parser.add_argument("--hyp_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--batch_size", type=int, default=1, help="物理温度控制参数B (目前是全批量梯度下降，为了模拟SGD热力学引入此虚拟参数)")
    parser.add_argument("--telm_every", type=int, default=10)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
        "--run_dir",
        type=str,
        default=str(_EXPERIMENT_ROOT / "runs" / "default_run"),
    )
    parser.add_argument("--c", type=float, default=1.0, help="庞加莱球曲率参数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full_matrix_loss",
        action="store_true",
        help="默认仅用上三角；指定则对全部 N^2 项求平均",
    )
    parser.add_argument(
        "--hf_online",
        action="store_true",
        help="允许联网加载 BGE（默认 HF 离线）",
    )
    parser.add_argument(
        "--synthetic_euclidean",
        action="store_true",
        help="跳过 BGE，使用合成欧氏嵌入（快速复现/无模型环境）",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.hf_online:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    train(
        num_nodes=args.num_nodes,
        hyp_dim=args.hyp_dim,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        telm_every=args.telm_every,
        data_path=args.data_path,
        run_dir=Path(args.run_dir),
        device=device,
        c=args.c,
        use_upper_triangle_only=not args.full_matrix_loss,
        seed=args.seed,
        force_synthetic=args.synthetic_euclidean,
    )


if __name__ == "__main__":
    main()
