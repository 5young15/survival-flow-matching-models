# 模型计算函数说明

本文档概括了生存分析模型的核心计算函数、输出内容及其对应的评价指标。

## 1. 核心计算函数 (统一接口)

所有模型均继承自 `SurvivalModelInterface`，并实现以下核心预测函数：

| 函数名称 | 输出内容 | 计算空间 | 对应评价指标 |
| :--- | :--- | :--- | :--- |
| `predict_risk` | 风险分数 $r(X)$ | 原始空间 | **C-index** |
| `predict_survival_function` | 生存概率 $S(t|X)$ | 原始空间 $[0, 1]$ | **IBS**, **Brier Score**, **Wasserstein-1** |
| `predict_time` | 预测时间（中位时间） | 原始空间 | **Median MAE**, **Median RMSE** |
| `compute_hazard_rate` | 对数风险函数 $\log h(t|X)$ | 对数空间 | **Hazard MSE**, **Hazard MAE**, **Hazard IAE** |

## 2. 扩展计算函数 (模型特有)

针对基于概率密度的模型（如 `FlowSurv`），提供以下扩展接口：

| 函数名称 | 输出内容 | 计算空间 | 对应评价指标 |
| :--- | :--- | :--- | :--- |
| `compute_log_density` | 对数概率密度 $\log f(t|X)$ | 对数空间 | **Density MSE**, **Density MAE** |

## 3. 指标与函数映射速查

| 评价指标 | 所需模型输出 | 推荐计算空间 |
| :--- | :--- | :--- |
| **C-index** | `predict_risk` | 原始空间 |
| **IBS / Brier Score** | `predict_survival_function` | 原始空间 |
| **MAE / RMSE** | `predict_time` | 原始空间 |
| **Hazard MSE / MAE** | `compute_hazard_rate` | **对数空间** |
| **Density MSE / MAE** | `compute_log_density` | **原始空间** (计算指标前进行 $\exp$ 转换) |
| **Wasserstein-1** | `predict_survival_function` | 原始空间 |

## 4. 数值稳定性截断

为保证计算稳定性，所有函数输出均遵循以下截断标准：

| 参数名称 | 截断值 | 应用范围 |
| :--- | :--- | :--- |
| $\epsilon_{log}$ | $10^{-8}$ | 对数运算截断 |
| $h_{max}$ | $1000$ | 风险函数上限 |
| $f_{max}$ | $1000$ | 密度函数上限 |
| $S_{min}$ | $10^{-100}$ | 生存函数下限 |
| $H_{max}$ | $100$ | 累积风险上限 |
