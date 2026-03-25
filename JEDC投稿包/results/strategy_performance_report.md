# Parisi Topological Risk Control - Strategy Performance Report
**Date:** 2026-01-12
**Methodology:** Walk-Forward Analysis (No Look-Ahead Bias)
**Risk Logic:** Tiered Risk Budgeting (100% -> 50% -> 0%)

---

## 1. Executive Summary

This report documents the performance of the **Topological Fragility Index (TFI)** based strategy after correcting for look-ahead bias and survivorship bias assumptions. 

The revised strategy employs a **Tiered Risk Control** mechanism:
- **Caution Phase (50% Exp):** When relative crowding > 80th percentile AND Price < MA20.
- **Crisis Phase (0% Exp):** When relative crowding > 95th percentile OR absolute crowding > 0.75.

---

## 2. S&P 100 Backtest Results (2000-2024)

| Metric | Benchmark (Buy & Hold) | TFI Strategy (Tiered) | Improvement |
| :--- | :--- | :--- | :--- |
| **Annualized Return** | 13.63% | **11.84%** | -1.79% |
| **Annualized Volatility** | 20.51% | **15.78%** | **-23.1%** (Lower Risk) |
| **Max Drawdown** | -52.15% | **-46.40%** | +5.75% (Better Defense) |
| **Sharpe Ratio** | 0.57 | **0.62** | **+8.8%** (Higher Efficiency) |

**Key Finding:** 
While total return is slightly lower due to "insurance premiums" (time out of market), the strategy significantly improves risk-adjusted returns (Sharpe Ratio) and reduces portfolio volatility by nearly a quarter.

---

## 3. China A-Share Backtest Results (2010-2024)

| Metric | Benchmark (Blue Chips) | TFI Strategy (Regime-Dependent) | Improvement |
| :--- | :--- | :--- | :--- |
| **Annualized Return** | 18.09% | **14.71%** | -3.38% |
| **Annualized Volatility** | 20.83% | **18.35%** | -11.9% (Lower Risk) |
| **Max Drawdown** | -34.70% | **-44.64%** | -9.94% (Worse Defense) |
| **Sharpe Ratio** | 0.75 | **0.67** | -10.6% |

**Key Finding:**
In the momentum-driven Emerging Market, the strategy struggled to outperform. The frequent de-risking during the 2017-2020 "Core Asset Bubble" led to underperformance. The failure to reduce Max Drawdown suggests that topological signals may lag in markets driven by exogenous policy shocks rather than endogenous liquidity dynamics. This serves as an important boundary condition for the theory.

---

## 4. Conclusion for Academic Submission

1.  **Validity**: The TFI indicator offers genuine **orthogonal information** that improves the Sharpe Ratio in mature, efficient markets (US).
2.  **Robustness**: The results are now robust to look-ahead bias, utilizing strict rolling windows for parameter estimation.
3.  **Limitations**: The strategy is less effective in markets with structural trading constraints (China), highlighting the importance of execution liquidity for topological risk control.
