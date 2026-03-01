# 模型计算函数文档

本文档记录每个生存分析模型的计算函数，包括：
- 计算什么数值
- 输出空间
- 用于计算什么指标

---

## 1. 接口基类 (interface.py)

### SurvivalModelInterface

| 函数 | 返回值 | 空间 | 用于指标 |
|------|--------|------|----------|
| `predict_risk(features)` | 风险分数 $r(X)$ | 原始空间 | C-index |
| `predict_survival_function(features, time_grid)` | 生存概率 $S(t\|X)$ | 原始空间 [0,1] | IBS, Brier Score |
| `predict_time(features, mode)` | 预测时间 | 原始空间 | Median MAE/RMSE |
| `compute_hazard_rate(features, time_grid)` | 风险函数 $\log h(t\|X)$ | 对数空间 | Hazard MSE/MAE/IAE |

---

## 2. FlowSurv (flowmodel/base_flow.py)

### 继承方法

| 函数 | 返回值 | 空间 | 用于指标 |
|------|--------|------|----------|
| `predict_risk(features)` | $-t_{median}$ | 原始空间 | C-index |
| `predict_survival_function(features, time_grid)` | $S(t\|X)$ | 原始空间 [0,1] | IBS, Brier Score |
| `predict_time(features, mode)` | 预测时间 | 原始空间 | Median MAE/RMSE |
| `compute_hazard_rate(features, time_grid)` | $\log h(t\|X)$ | 对数空间 | Hazard MSE/MAE/IAE |

### 特有方法

#### predict_survival_metrics(features, time_grid, ode_steps)
计算完整生存指标集，返回字典：

| 键 | 返回值 | 空间 | 用于指标 |
|----|--------|------|----------|
| `'log_hazard'` | $\log h(t\|X)$ | 对数空间 | Hazard MSE/MAE/IAE |
| `'log_density'` | $\log f(t\|X)$ | 对数空间 | Density MSE/MAE |
| `'survival'` | $S(t\|X)$ | 原始空间 [0,1] | IBS, Brier Score, Wasserstein-1 |
| `'cum_hazard'` | $H(t\|X)$ | 原始空间 | - |
| `'log_survival'` | $\log S(t\|X)$ | 对数空间 | - |

#### compute_density(features, time_grid, ode_steps)
| 返回值 | 空间 | 用于指标 |
|--------|------|----------|
| $\log f(t\|X)$ | 对数空间 | Density MSE/MAE |

---

## 3. GumbelFlowSurv (flowmodel/gumbel_flow.py)

### 继承方法
与 FlowSurv 相同

### 特有方法

#### compute_density(features, time_grid, ode_steps)
| 返回值 | 空间 | 用于指标 |
|--------|------|----------|
| $\log f(t\|X)$ | 对数空间 | Density MSE/MAE |

---

## 4. LinearCoxPH (baselines/coxph.py)

### 继承方法

| 函数 | 返回值 | 空间 | 用于指标 |
|------|--------|------|----------|
| `predict_risk(features)` | $\beta^T X$ (log-HR) | 原始空间 | C-index |
| `predict_survival_function(features, time_grid)` | $S(t\|X) = \exp(-H_0(t) \cdot e^{\beta^T X})$ | 原始空间 [0,1] | IBS, Brier Score |
| `predict_time(features, mode)` | 中位生存时间 | 原始空间 | Median MAE/RMSE |
| `compute_hazard_rate(features, time_grid)` | $\log h(t\|X) = \log h_0(t) + \beta^T X$ | 对数空间 | Hazard MSE/MAE/IAE |

---

## 5. DeepSurv (baselines/deepsurv.py)

### 继承方法

| 函数 | 返回值 | 空间 | 用于指标 |
|------|--------|------|----------|
| `predict_risk(features)` | $\text{MLP}(X)$ (log-HR) | 原始空间 | C-index |
| `predict_survival_function(features, time_grid)` | $S(t\|X)$ | 原始空间 [0,1] | IBS, Brier Score |
| `predict_time(features, mode)` | 中位生存时间 | 原始空间 | Median MAE/RMSE |
| `compute_hazard_rate(features, time_grid)` | $\log h(t\|X)$ | 对数空间 | Hazard MSE/MAE/IAE |

---

## 6. DeepHit (baselines/deephit.py)

### 继承方法

| 函数 | 返回值 | 空间 | 用于指标 |
|------|--------|------|----------|
| `predict_risk(features)` | $-\mathbb{E}[T]$ | 原始空间 | C-index |
| `predict_survival_function(features, time_grid)` | $S(t\|X) = 1 - \sum_{k: t_k \leq t} p_k$ | 原始空间 [0,1] | IBS, Brier Score |
| `predict_time(features, mode)` | 中位生存时间 | 原始空间 | Median MAE/RMSE |
| `compute_hazard_rate(features, time_grid)` | $\log h(t\|X) = \log(f(t)/S(t))$ | 对数空间 | Hazard MSE/MAE/IAE |

### 特有方法

#### compute_density(features, time_grid)
DeepHit 不提供密度函数

---

## 7. WeibullAFT (baselines/weibullAFT.py)

### 继承方法

| 函数 | 返回值 | 空间 | 用于指标 |
|------|--------|------|----------|
| `predict_risk(features)` | $-t_{median}$ | 原始空间 | C-index |
| `predict_survival_function(features, time_grid)` | $S(t\|X) = \exp(-(t/\lambda)^k)$ | 原始空间 [0,1] | IBS, Brier Score |
| `predict_time(features, mode)` | 中位生存时间 $t_{med} = \lambda \cdot (\ln 2)^{1/k}$ | 原始空间 | Median MAE/RMSE |
| `compute_hazard_rate(features, time_grid)` | $\log h(t\|X) = \log(k/\lambda) + (k-1)\log(t/\lambda)$ | 对数空间 | Hazard MSE/MAE/IAE |

---

## 8. RSF (baselines/RSF.py)

### 继承方法

| 函数 | 返回值 | 空间 | 用于指标 |
|------|--------|------|----------|
| `predict_risk(features, time_grid)` | 风险分数 | 原始空间 | C-index |
| `predict_survival_function(features, time_grid)` | $S(t\|X)$ | 原始空间 [0,1] | IBS, Brier Score |
| `predict_time(features)` | 中位生存时间 | 原始空间 | Median MAE/rmse |
| `compute_hazard_rate(features, time_grid)` | $\log h(t\|X)$ | 对数空间 | Hazard MSE/MAE/IAE |

---

## 9. 数据生成器 (experiments/data_generation.py)

### SurvivalData

| 属性 | 值 | 空间 | 用于指标 |
|------|-----|------|----------|
| `true_hazard` | $h(t\|X)$ | 原始空间 | Hazard MSE/MAE/IAE |
| `true_density` | $f(t\|X)$ | 原始空间 | Density MSE/MAE |
| `true_survival` | $S(t\|X)$ | 原始空间 [0,1] | Wasserstein-1 |
| `true_medians` | $t_{median}$ | 原始空间 | Median MAE/RMSE |

---

## 10. 计算指标 (experiments/metrics.py)

| 指标函数 | 输入空间 | 输出 | 说明 |
|----------|----------|------|------|
| `concordance_index_pytorch` | 原始空间 | [0,1] | 风险排序一致性 |
| `integrated_brier_score` | 原始空间 | [0,1] | 整体生存预测误差 |
| `time_dependent_brier_score` | 原始空间 | [0,1] | 指定时间点误差 |
| `median_time_error` | 原始空间 | $\mathbb{R}^+$ | 中位时间误差 |
| `hazard_mse` | 对数空间 | $\mathbb{R}^+$ | 对数风险函数 MSE |
| `hazard_mae` | 对数空间 | $\mathbb{R}^+$ | 对数风险函数 MAE |
| `hazard_integrated_absolute_error` | 对数空间 | $\mathbb{R}^+$ | 对数风险函数积分误差 |
| `density_mse` | 对数空间 | $\mathbb{R}^+$ | 对数密度函数 MSE |
| `density_mae` | 对数空间 | $\mathbb{R}^+$ | 对数密度函数 MAE |
| `wasserstein_1_distance` | 对数空间 | $\mathbb{R}^+$ | 对数 CDF 距离 |

---

## 11. 空间转换流程

```
数据生成器 (原始空间)
        ↓
    模型预测
        ↓
┌───────┴───────┐
│              │
原始空间       对数空间
(S, risk)    (log_h, log_f)
    ↓            ↓
    └────┬────┘
         ↓
    计算指标
         ↓
   MetricsResult
```

---

## 12. 数值稳定性截断

| 参数 | 值 | 用途 |
|------|-----|------|
| $\epsilon_{log}$ | $10^{-8}$ | 对数空间截断 |
| $\epsilon_{density}$ | $10^{-100}$ | 密度截断 |
| $\epsilon_{survival}$ | $10^{-100}$ | 生存函数截断 |
| $h_{max}$ | $1000$ | 风险函数上限 |
| $f_{max}$ | $1000$ | 密度函数上限 |
| $H_{max}$ | $100$ | 累积风险上限 |

---

## 13. 修改记录

| 日期 | 修改内容 |
|------|----------|
| 2024-03-01 | 初始文档 |
