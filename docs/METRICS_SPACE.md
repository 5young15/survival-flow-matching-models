# 生存分析指标说明

| 指标名称 | 计算空间 | 计算公式 | 中文说明 |
| :--- | :--- | :--- | :--- |
| **C-index** | 原始空间 | $\frac{\sum \mathbb{1}(T_i < T_j, \delta_i=1) \cdot (\mathbb{1}(r_i > r_j) + 0.5 \cdot \mathbb{1}(r_i = r_j))}{\sum \mathbb{1}(T_i < T_j, \delta_i=1)}$ | 一致性指数，衡量风险分数 $r$ 与生存时间的排序一致性，包含对相等风险的处理。 |
| **IBS** | 原始空间 | $\frac{1}{T_{max} - T_{min}} \int_{T_{min}}^{T_{max}} BS(t) dt$ | 积分 Brier Score，在整个时间范围内对生存概率预测误差的加权积分。 |
| **Brier Score** | 原始空间 | $\frac{1}{N} \sum w_i(t) \cdot (y_i(t) - \hat{S}(t|X_i))^2$ | 给定时间点的概率预测误差，使用 IPCW 权重 $w_i(t)$ 处理数据删失。 |
| **Median MAE** | 原始空间 | $\mathbb{E}[|t_{med}^{true} - t_{med}^{pred}|]$ | 中位生存时间的平均绝对误差，衡量预测时间与真实时间的偏差。 |
| **Hazard MSE** | 对数空间 | $\mathbb{E}[(\log h_{true} - \log h_{pred})^2]$ | 风险函数的均方误差。在对数空间计算以处理跨越多个数量级的数值差异。 |
| **Hazard IAE** | 对数空间 | $\int |\log h_{true}(t) - \log h_{pred}(t)| dt$ | 风险函数的积分绝对误差。衡量整个生存期内风险率预测的累积偏差。 |
| **Density MSE** | 原始空间 | $\mathbb{E}[(f_{true} - f_{pred})^2]$ | 概率密度函数的均方误差。密度函数 $f(t)$ 满足归一化，直接在原始空间比较。 |
| **Wasserstein-1** | 对数空间 | $\int |\log(1 - S_{true}) - \log(1 - S_{pred})| dt$ | 衡量生存函数分布差异。在对数空间（$\log F$）计算，可有效放大尾部区域的预测误差。 |

---
**注：** 
- **原始空间**：适用于概率值 $[0, 1]$、时间值或排序相关指标。
- **对数空间**：适用于数值跨度极大（如 Hazard）或需增强尾部敏感度（如 Wasserstein）的指标。
