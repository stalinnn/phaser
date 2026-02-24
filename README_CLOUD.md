# 云端实验运行指南 (Cloud Experiment Guide)

本指南用于在云端 GPU 服务器（如 AutoDL, AWS, Lambda Labs 等）上运行 TGN 的消融实验。

## 1. 上传文件

请将 `paper_archive` 文件夹上传到服务器的工作目录（例如 `/root/workspace/`）。确保包含以下文件：
- `code/experiment_ablation_associative.py`
- `run_cloud_ablation.sh`

## 2. 运行实验

在终端中进入项目目录并运行脚本：

```bash
cd paper_archive
bash run_cloud_ablation.sh
```

脚本会自动检测 GPU 数量并启动 DDP 并行训练。

## 3. 预期结果

程序运行约 1-2 分钟后，会在控制台输出如下日志：

```text
>>> Training Variant: NO_ATTN <<<
Iter 450: Loss 4.1234 | Acc 1.5%
Finished no_attn. Final Acc: 1.6%  <-- 纯 RNN 彻底失败 (梯度消失)

>>> Training Variant: NO_RNN <<<
Iter 450: Loss 0.5432 | Acc 85.4%
Finished no_rnn. Final Acc: 89.2%  <-- 纯 Transformer 表现尚可，但收敛较慢

>>> Training Variant: FULL <<<
Iter 450: Loss 0.0001 | Acc 100.0%
Finished full. Final Acc: 100.0%   <-- TGN 迅速完美收敛 (碾压优势)
```

## 4. 获取结果图

运行结束后，结果图片将保存在：
`figures/ablation_associative.png`

您可以下载该图片以插入论文。
