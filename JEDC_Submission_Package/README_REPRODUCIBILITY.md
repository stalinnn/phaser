# JEDC 投稿论文复现指南 (Reproducibility Guide)

本文件详细记录了论文《尾部风险的微观基础：基于逆向注意力机制的市场拓扑重构》中核心数值结论、图表及统计检验的复现代码路径。

> **论文版本**: Paper_Final_CN.md (2026-01-12)
> **代码环境**: Python 3.8+ (依赖: pandas, numpy, scipy, sklearn, matplotlib, yfinance, statsmodels)

---

## 1. 快速开始 (Quick Start)

为了方便审稿人快速验证，我们提供了一个一键运行脚本，它将按顺序执行所有核心分析并生成图表：

```bash
python code/run_all_reproduction.py
```

---

## 2. 核心数值结论复现 (Key Numerical Results)

### A. 理论验证：微观基础模拟 (Micro-foundations)
*   **对应章节**: Section 2.3 & Figure 2A
*   **复现命令**: `python code/run_abm_simulation.py`
*   **功能**: 运行基于主体模型 (Agent-Based Model)，模拟羊群效应 (Herding) 如何导致 TFI 序参量飙升。
*   **输出**: 生成 `figures/abm_simulation_proof.png`。

### B. 计量检验：峰度预测 (Econometric Test)
*   **对应章节**: Section 3.2 (Table 1)
*   **复现命令**: `python code/run_asset_pricing_test.py`
*   **核心结论**: 在控制 VIX 后，TFI 指标依然显著预测未来 20 天的**峰度 (Kurtosis)** ($t=2.42, p=0.015$)。
*   **输出**: 控制台打印 OLS 回归表，并保存至 `results/kurtosis_regression.txt`。

### C. 美股策略回测 (US Strategy Performance)
*   **对应章节**: Section 4.2 (Table 2)
*   **复现命令**: `python code/backtest_parisi_strategy.py`
*   **策略逻辑**: **分级风险预算 (Tiered Risk Control)** (100% -> 50% -> 0%)，基于动态滚动分位数。
*   **数据增强**: 该脚本现在使用 `code/data_provider.py` 模块，支持导入外部 Point-in-Time 数据以消除生存偏差。
*   **核心结论**:
    *   年化波动率：20.51% -> **15.78%** (-23%)
    *   夏普比率：0.57 -> **0.62** (+8.8%)
*   **输出**: 生成 `figures/backtest_enhanced_data.png`。

### D. 中国市场回测 (China Market Check)
*   **对应章节**: Appendix A.1
*   **复现命令**: `python code/backtest_china_a_share.py`
*   **策略逻辑**: **体制依赖型风控 (Regime-Dependent)**，基于 MA60 牛熊分界。
*   **核心结论**: 策略在 A 股跑输基准，证明了拓扑指标在政策驱动型市场中的局限性。
*   **输出**: 生成 `figures/china_market_corrected.png`。

### E. 竞品对比 (Benchmark Comparison)
*   **对应章节**: Section 3.1 & Table 3
*   **复现命令**: `python code/benchmark_comparison_test.py`
*   **核心结论**: TFI 的召回率 (91.8%) 显著高于 Absorption Ratio (63.0%)，准确率 (6.9%) 略优于 AR (6.1%)，且在美股危机中平均提前 4.5 天报警。
*   **输出**: 控制台打印详细的 Precision/Recall 表格及 Lead Time 统计。

---

## 3. 论文图表对应关系 (Figures Mapping)

| 论文图表编号 | 图表名称 | 对应生成脚本 | 备注 |
| :--- | :--- | :--- | :--- |
| **Figure 1** | **正交性与背离** | `code/core_model.py` | 验证 TFI 与平均相关性的背离。生成 `figures/tier1_v2_result.png`。 |
| **Figure 1C**| **系统性风险基准对比** | `code/plot_tfi_vs_benchmark.py` | **[新增]** 对比 TFI 与 Absorption Ratio (Kritzman 2010)。 |
| **Figure 2A** | **微观机制模拟** | `code/run_abm_simulation.py` | **[新增]** ABM 模拟结果，展示相变机制。 |
| **Figure 3** | **参数鲁棒性** | `code/china_robustness_test.py` | (原 Figure 2) 参数敏感性热力图。 |
| **Figure 4** | **分级策略回测** | `code/backtest_parisi_strategy.py` | (原 Figure 3) 美股 Walk-Forward 回测曲线。 |
| **Figure 5** | **2020 熔断案例** | `code/plot_lead_time_event_study.py` | (原 Figure 4) 2020 年疫情期间的信号细节。 |
| **Figure A1** | **中国市场回测** | `code/backtest_china_a_share.py` | 修正后的 A 股策略表现（无未来函数）。 |
| **Robustness**| **安慰剂检验 (Placebo)** | `code/run_placebo_test.py` | **[新增]** 验证 TFI 并非随机噪声 (KS Test)。 |

---

## 4. 数据说明与生存偏差处理 (Data & Survivorship Bias)

*   **数据源**: 所有脚本默认通过 `yfinance` 获取数据。
*   **局限性声明**: 公共 API 通常缺乏已退市股票的历史数据。
*   **改进机制**: 本项目引入了 `code/data_provider.py` 模块：
    1.  内置了扩展的股票池（包含 LEH, BSC 等历史退市股），以便在数据可用时自动纳入。
    2.  支持优先读取本地清洗后的数据文件 (`code/data/market_data_cleaned.csv`)。
    3.  详细的数据准备说明请参考: `code/data/DATA_PREPARATION_GUIDE.md`。

---
*Document updated for JEDC Submission (2026-01-12)*
