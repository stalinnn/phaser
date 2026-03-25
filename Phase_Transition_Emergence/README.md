# 全息相变涌现训练 (Phase Transition Emergence)

本目录实现「边界欧氏语义相似度 → 体空间双曲度规对齐」的离散 RT 映射实验，并记录 TELM 探针（有效秩、梯度范数等），用于观察训练过程中的结构涌现与可能的相变迹象。

## 依赖

- `torch`
- `geoopt`
- `numpy`
- `sentence-transformers`（可选；不可用时自动用合成欧氏嵌入）
- `matplotlib`（仅 `visualize_phase.py`）

## 运行训练

```bash
cd /phaser/Phase_Transition_Emergence
python emerge_holographic_bulk.py \
  --num_nodes 128 \
  --hyp_dim 16 \
  --epochs 400 \
  --lr 0.08 \
  --run_dir runs/demo1
```

可选：自定义文本（每行一段或 `.jsonl` 含 `text` 字段）：

```bash
python emerge_holographic_bulk.py --data_path data/corpus.txt --run_dir runs/with_text
```

若本机已有 BGE-M3 权重，默认 `HF_HUB_OFFLINE=1`；需联网拉模型时加 `--hf_online`。

## 可视化

```bash
python visualize_phase.py --run_dir runs/demo1
```

生成：

- `phase_transition_curves.png`：Loss / 有效秩 / 梯度范数 / 双曲范数均值
- `hyperbolic_embedding_pca2d.png`：双曲嵌入的 PCA 二维投影

## 目录说明

| 路径 | 说明 |
|------|------|
| `core/hyperbolic_space.py` | 庞加莱球、两两测地线距离 |
| `core/metric_alignment.py` | 余弦矩阵、纠缠代理、度规对齐损失 |
| `probes/telm_monitor.py` | 有效秩与 TELM 读数 |
| `emerge_holographic_bulk.py` | 主训练脚本 |
| `visualize_phase.py` | 结果绘图 |
| `data/` | 可选语料放置处 |

## 理论备注

损失形式：在目标矩阵 \(T\)（由边界余弦相似度映射到 \([0,1]\)）与 \(\exp(-d_{\mathbb{H}})\) 之间做 MSE，使高语义相似在体空间对应更短测地线。训练仅更新双曲坐标，欧氏嵌入固定为「边界 CFT」观测。
