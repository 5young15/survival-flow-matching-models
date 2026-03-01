# 生存分析指标计算空间文档

## 1. 指标空间总览

| 指标 | 计算空间 | 公式 | 说明 |
|------|----------|------|------|
| **C-index** | 原始空间 | $\frac{\text{concordant} + 0.5 \times \text{tied}}{\text{permissible}}$ | 风险分数排序，对数变换不影响 |
| **IBS** | 原始空间 | $\frac{1}{T_{max}-T_{min}}\int_{T_{min}}^{T_{max}} BS(t) dt$ | 概率值 ∈ [0,1] |
| **Brier Score** | 原始空间 | $\frac{1}{N}\sum (y(t) - \hat{S}(t\|X))^2$ | 概率值 ∈ [0,1] |
| **Median MAE** | 原始空间 | $\mathbb{E}[|t_{med}^{true} - t_{med}^{pred}|]$ | 时间值需保持原始尺度 |
| **Median RMSE** | 原始空间 | $\sqrt{\mathbb{E}[(t_{med}^{true} - t_{med}^{pred})^2]}$ | 时间值需保持原始尺度 |
| **Hazard MSE** | 对数空间 | $\mathbb{E}[(\log h_{true} - \log h_{pred})^2]$ | 处理跨数量级差异 |
| **Hazard MAE** | 对数空间 | $\mathbb{E}[\|\log h_{true} - \log h_{pred}\|]$ | 处理跨数量级差异 |
| **Hazard IAE** | 对数空间 | $\int\|\log h_{true}(t) - \log h_{pred}(t)\|dt$ | 处理跨数量级差异 |
| **Density MSE** | 原始空间 | $\mathbb{E}[(f_{true} - f_{pred})^2]$ | 概率密度值直接比较 |
| **Density MAE** | 原始空间 | $\mathbb{E}[|f_{true} - f_{pred}|]$ | 概率密度值直接比较 |
| **Wasserstein-1** | 对数空间 | $\int \|\log(1-S_{true}) - \log(1-S_{pred})\|dt$ | 放大尾部差异 |

---

## 2. 核心函数定义

### 2.1 生存函数 (Survival Function)
$$S(t|X) = P(T > t | X)$$

- 表示在时间 $t$ 后事件未发生的概率
- 值域: $[0, 1]$
- 性质: $S(0) = 1$, $\lim_{t \to \infty} S(t) = 0$

### 2.2 风险函数 (Hazard Function)
$$h(t|X) = \lim_{\Delta t \to 0} \frac{P(t < T \leq t + \Delta t | T > t)}{\Delta t} = \frac{f(t|X)}{S(t|X)}$$

- 条件瞬时事件发生率
- 可能跨越多个数量级 ($10^{-6}$ ~ $10^6$)
- 对数空间计算优势:
  - 原始空间 MSE: $\mathbb{E}[(h_{true} - h_{pred})^2]$
  - 对数空间 MSE: $\mathbb{E}[(\log h_{true} - \log h_{pred})^2]$

### 2.3 密度函数 (Probability Density Function)
$$f(t|X) = -\frac{d}{dt}S(t|X) = h(t|X) \cdot S(t|X)$$

- 事件时间分布的密度
- 满足归一化: $\int_0^{\infty} f(t|X) dt = 1$
- 可能跨越多个数量级

### 2.4 累积风险函数 (Cumulative Hazard)
$$H(t|X) = \int_0^t h(u|X) du = -\log S(t|X)$$

- 风险函数的积分
- 值域: $[0, \infty)$

### 2.5 累积分布函数 (CDF)
$$F(t|X) = 1 - S(t|X) = \int_0^t f(u|X) du$$

- 事件发生的累积概率

---

## 3. 各指标详细公式

### 3.1 C-index (一致性指数)
$$C = \frac{\sum_{i,j} \mathbb{1}(T_i < T_j, \delta_i=1) \cdot \mathbb{1}(r_i > r_j)}{\sum_{i,j} \mathbb{1}(T_i < T_j, \delta_i=1)}$$

其中:
- $T$: 事件时间
- $\delta$: 事件指示器 (1=事件发生, 0=删失)
- $r$: 风险分数

### 3.2 IBS (积分 Brier Score)
$$IBS = \frac{1}{T_{max} - T_{min}} \int_{T_{min}}^{T_{max}} BS(t) dt$$

$$BS(t) = \frac{1}{N} \sum_{i=1}^N w_i(t) \cdot (y_i(t) - \hat{S}(t|X_i))^2$$

其中 IPCW 权重:
$$w_i(t) = \frac{\mathbb{1}(T_i \leq t, \delta_i=1)}{G(T_i)} + \frac{\mathbb{1}(T_i > t)}{G(t)}$$

$G(t)$ 是删失分布的 Kaplan-Meier 估计

### 3.3 Hazard MSE/MAE (对数空间)
$$MSE_{log\_h} = \frac{1}{N}\sum_{i=1}^N \frac{1}{T}\sum_{t \in T} (\log h_i^{true}(t) - \log h_i^{pred}(t))^2$$

$$MAE_{log\_h} = \frac{1}{N}\sum_{i=1}^N \frac{1}{T}\sum_{t \in T} |\log h_i^{true}(t) - \log h_i^{pred}(t)|$$

### 3.4 Hazard IAE (对数空间积分绝对误差)
$$IAE_{log\_h} = \frac{1}{N}\sum_{i=1}^N \int_{T_{min}}^{T_{max}} |\log h_i^{true}(t) - \log h_i^{pred}(t)| dt$$

使用梯形积分近似:
$$\int f(t) dt \approx \sum_{k=1}^{T-1} \frac{f(t_k) + f(t_{k+1})}{2} \cdot (t_{k+1} - t_k)$$

### 3.5 Density MSE/MAE (原始空间)
$$MSE_{f} = \frac{1}{N}\sum_{i=1}^N \frac{1}{T}\sum_{t \in T} (f_i^{true}(t) - f_i^{pred}(t))^2$$

$$MAE_{f} = \frac{1}{N}\sum_{i=1}^N \frac{1}{T}\sum_{t \in T} |f_i^{true}(t) - f_i^{pred}(t)|$$

**原始空间计算原因:**
- 密度函数 $f(t)$ 满足归一化：$\int_0^{\infty} f(t) dt = 1$
- 密度值通常在有限范围内，不会跨越多个数量级
- 原始空间直接比较概率密度更符合直观解释

### 3.6 Wasserstein-1 距离 (对数空间)
$$W_1 = \frac{1}{N}\sum_{i=1}^N \int_{T_{min}}^{T_{max}} |\log(1 - S_i^{true}(t)) - \log(1 - S_i^{pred}(t))| dt$$

**公式推导:**
- 原始空间: $W_1 = \int |F_{true}(t) - F_{pred}(t)| dt$，其中 $F = 1 - S$
- 对数空间: $\log F = \log(1 - S)$

**对数空间优势:**
- 当 $S \to 1$ 时，$F \to 0$，原始空间差异极小
- 对数空间放大尾部差异：$\log F$ 在 $F \to 0$ 时趋向 $-\infty$
- 更敏感地捕捉生存曲线尾部预测误差

---

## 4. 各模型接口返回值

### Flow 模型 (base_flow.py, gumbel_flow.py)

| 方法 | 返回值 | 空间 |
|------|--------|------|
| `predict_survival_metrics()['log_hazard']` | $\log h(t\|X)$ | 对数 |
| `predict_survival_metrics()['log_density']` | $\log f(t\|X)$ | 对数 |
| `predict_survival_metrics()['survival']` | $S(t\|X)$ | 原始 |
| `predict_survival_metrics()['cum_hazard']` | $H(t\|X)$ | 原始 |
| `predict_survival_metrics()['log_survival']` | $\log S(t\|X)$ | 对数 |
| `compute_hazard_rate()` | $\log h(t\|X)$ | 对数 |
| `compute_log_density()` | $\log f(t\|X)$ | 对数 |

### 基线模型 (CoxPH, DeepSurv, DeepHit, WeibullAFT)

| 方法 | 返回值 | 空间 |
|------|--------|------|
| `predict_survival_function()` | $S(t\|X)$ | 原始 |
| `predict_risk()` | $r(X)$ | 原始 |
| `predict_time()` | $t_{med}$ | 原始 |
| `compute_hazard_rate()` | $\log h(t\|X)$ | 对数 |
| `compute_density()` | 无 | - |

---

## 5. 空间转换

### 5.1 转换公式

**原始空间 → 对数空间:**
```
log_h = log(h_raw + ε)      # 风险函数
log_f = log(f_raw + ε)      # 密度函数
log_F = log(1 - S + ε)      # CDF 的对数 (用于 Wasserstein)
```

**对数空间 → 原始空间:**
```
h_raw = exp(log_h) - ε      # 风险函数
f_raw = exp(log_f) - ε      # 密度函数
S = 1 - exp(log_F) + ε      # 生存函数
```

其中 $\epsilon = 10^{-8}$ 为数值稳定性截断，防止 $\log(0)$

### 5.2 关键关系

$$\log h(t) = \log f(t) - \log S(t)$$

$$H(t) = -\log S(t) \quad \text{(累积风险)}$$

### 5.3 空间选择原则

| 场景 | 推荐空间 |
|------|----------|
| 概率值 [0,1] | 原始空间 |
| 排序相关指标 | 原始空间 |
| 多数量级差异 | 对数空间 |
| 尾部区域敏感度 | 对数空间 |

---

## 6. 修改记录

| 日期 | 修改内容 |
|------|----------|
| 2025-03-01 | Density MSE/MAE 改为原始空间计算，其他指标保持不变 |
| 2025-03-01 | GumbelFlowSurv 删除冗余的 `predict_risk` 方法，复用父类 FlowSurv 实现 |
| 2025-03-01 | 统一所有模型接口，删除冗余方法 |
| 2025-03-01 | Cox 模型复用父类 `_cox_predict_time`, `_cox_compute_hazard_rate` |
| 2025-03-01 | Flow 模型删除冗余的 `compute_density`，统一使用 `compute_log_density` |
| 2025-03-01 | metrics.py 简化空间检测逻辑，统一输入为对数空间 |
| 2025-03-01 | 修复 Median MAE 公式括号错误 |
| 2025-03-01 | 完善 Wasserstein-1 公式推导说明 |
| 2025-03-01 | 补充空间转换关键关系公式 |
| 2024-03-01 | 统一所有模型的 hazard/density 返回对数空间 |
| 2024-03-01 | Wasserstein-1 距离改为对数空间计算 |
| 2024-03-01 | 添加详细数学公式文档 |
