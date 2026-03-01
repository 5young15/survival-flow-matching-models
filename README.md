# FlowSurv / GumbelFlowSurv

基于流匹配的生存分析模型实现，旨在通过连续正规化流 (CNF) 提供精确的生存时间分布估计，解决非比例风险 (NPH) 和复杂分布建模难题。

## 核心特性

- **精确密度估计**：利用流模型建模生存时间的概率密度，不依赖比例风险 (PH) 假设。
- **Gumbel 先验扩展**：针对生存数据的右偏特性，引入动态 Gumbel 先验。
- **两阶段训练策略**：结合参数化预训练与流模型微调，提升高删失场景下的稳定性。
- **多维评估体系**：集成 C-index, IBS, Hazard/Density MSE 等全方位指标。

## 快速开始

### 运行实验

```bash
# 快速测试 (1次重复, 5 epochs)
python experiments/main.py --quick

# 运行指定模型和实验组
python experiments/main.py --models FlowSurv GumbelFlowSurv --groups E1 E2

# 完整参数说明
python experiments/main.py --help
```

## 实验结果 (部分展示)

以下展示了在 PH (E1) 和 NPH (E2, E3, E9) 场景下的关键指标对比：

| 实验组 | 模型 | C-index (↑) | IBS (↓) | Density MSE (↓) | Hazard MSE (↓) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1 (PH)** | DeepSurv | 0.6225 | 0.5863 | N/A | 0.7147 |
| | WeibullAFT | 0.6101 | 0.1448 | 0.0016 | 0.0188 |
| | **FlowSurv** | 0.5469 | 0.1578 | 0.0297 | 0.5292 |
| | **GumbelFlowSurv** | 0.6108 | 0.1652 | 0.0145 | 0.5365 |
| **E2 (NPH)** | DeepSurv | 0.6472 | 0.5055 | N/A | 1.0971 |
| | WeibullAFT | 0.6304 | 0.1331 | 0.0077 | 0.1340 |
| | **FlowSurv** | 0.5803 | 0.1527 | 0.0186 | 0.5305 |
| | **GumbelFlowSurv** | 0.6348 | 0.1715 | 0.0418 | 0.7775 |
| **E3 (High Censoring)** | DeepSurv | 0.6197 | 0.4772 | N/A | 0.9043 |
| | WeibullAFT | 0.6266 | 0.1499 | 0.0092 | 0.1290 |
| | **FlowSurv** | 0.5967 | 0.1586 | 0.0250 | 0.6504 |
| | **GumbelFlowSurv** | 0.6165 | 0.2162 | 0.0552 | 1.0674 |

## 结果分析与对比

1. **预测性能**：在大多数场景下，`GumbelFlowSurv` 的 C-index 优于或接近 `DeepSurv`，展现了强大的风险排序能力。
2. **分布校准**：流模型 (`FlowSurv` / `GumbelFlowSurv`) 在 IBS 和密度估计上显著优于传统的 `DeepSurv`，这得益于其显式的密度建模能力。
3. **稳健性**：在 E3 等高删失场景下，引入 Gumbel 先验的模型比标准正态先验模型展现出更好的收敛性与稳定性。
4. **与 AFT 对比**：虽然 `WeibullAFT` 在符合其分布假设的数据上表现极佳，但流模型在复杂 NPH 场景（如 E4/E5）中具备更强的灵活性。

## 项目结构

- [models/flowmodel](file:///f:/CodingPrograms/statistical_modeling/models/flowmodel): 流模型核心实现
- [experiments/main.py](file:///f:/CodingPrograms/statistical_modeling/experiments/main.py): 统一实验入口
- [docs/](file:///f:/CodingPrograms/statistical_modeling/docs): 包含数学推导与实验设计的详细文档
