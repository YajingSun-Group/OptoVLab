# OLED-GAT 可解释性研究报告

## 基于图注意力、输出梯度与层参数反事实的联合分析

更新时间：2026-07-27

## 摘要

本报告研究当前 OLED-GAT 模型能否从注意力机制出发，解释 OLED 多层器件
结构中不同功能层及层参数对最大外量子效率（EQEmax）预测的影响。

本次不是概念性讨论，而是直接分析冻结模型及冻结测试集：

- 模型：4 个残差 `GATv2Conv` 模块，每个模块 4 个注意力头。
- 测试集：423 个器件，来自 345 篇论文。
- 层节点：3,091 个。
- 有厚度数据的层：2,560 个。
- 完成上下 10% 厚度反事实的层：2,319 个。
- 解释目标：OLED-GAT 条件均值头输出的 EQEmax。

核心结论如下。

1. **模型把最多的输出相关注意力分配给 EML 及其直接边界。**
   在“器件含有该层”的条件下，按论文聚类统计的注意力-梯度份额中位数为：
   EBL 16.45%、EML 16.21%、HTL 14.99%、EIL 13.49%、anode 13.25%。

2. **深层消息传递主要聚焦于指向 EML 的界面信息。**
   第 4 个 GATv2 模块中，最强的界面消息依次包括
   `HTL -> EML`、`EBL -> EML`、`HIL -> EML`、`HBL -> EML`
   和 `ETL -> EML`。这与载流子平衡、激子形成和激子限制的 OLED
   物理图像相容，但只能称为“模型行为与物理先验一致”，不能称为因果证明。

3. **原始注意力只有中等程度的忠实性。**
   原始注意力与删除对应层-根节点连接后预测变化的总体 Spearman 相关系数为
   0.503；器件内部相关系数中位数为 0.536。该强度不足以把注意力权重直接解释为
   “该层对 EQE 的贡献率”。

4. **不同注意力头的排序并不稳定。**
   四个模块的头间层排序相关系数中位数分别为
   0.053、0.133、0.524 和 -0.083。多个头明显学习了不同的信息路由模式；
   简单平均注意力会掩盖这种差异。

5. **当前模型对层厚度的数值敏感性很弱。**
   EML 厚度上下变化 10% 时，模型预测变化绝对值的论文聚类中位数仅为
   0.0174 个 EQE 百分点；其他主要层通常低于 0.003 个百分点。
   因而当前模型不能支持“某层增厚多少会显著提高 EQE”这样的定量结论。

6. **当前模型主要通过层角色、材料身份、EML 分子结构和器件家族关联预测，
   而不是通过完整的器件物理参数预测。**
   这是当前可解释性分析最重要的模型诊断。

7. **当前最佳最终模型是 62% CatBoost 与 38% OLED-GAT 的集成。**
   本报告中的注意力只解释 OLED-GAT 分量，不能代表完整集成预测。

综合判断：

> 注意力适合用于定位模型关注的层和界面，并形成实验假设；层参数的方向和幅度
> 必须由梯度、受控反事实、同论文匹配器件和最终实验共同验证。

---

## 1. 研究问题

本次可解释性研究需要回答四个层次的问题。

### 1.1 局部解释

对于某一个具体器件：

- 哪些功能层最影响该器件的预测？
- 模型更关注 EML、传输层、阻挡层还是电极？
- 哪个相邻层界面承担了主要的信息传递？
- 某一层厚度改变后，预测 EQE 向什么方向变化、变化多大？

### 1.2 全局解释

在整个数据集上：

- 哪些层角色通常获得较高注意力？
- 不同 GATv2 深度是否学习了不同物理尺度的信息？
- 注意力头之间是否形成稳定、可重复的分工？
- 层厚度效应是否在不同器件中方向一致？

### 1.3 物理解释

模型关注的层和界面是否与已知 OLED 机制相容：

- 载流子注入和传输平衡；
- EML 内电子与空穴复合；
- 电子、空穴及激子的限制；
- 发光层 PLQY 与激子利用率；
- 光学微腔与出光耦合。

OLED 的 EQE 通常可分解为电荷平衡、辐射激子形成、发光量子产率和出光效率等
乘积因素。已有 OLED 研究明确指出，输运层、阻挡层及其材料选择会通过电荷平衡
影响 EQE，而层厚度还可能通过输运和光学微腔共同作用
（[Scientific Reports, 2017](https://www.nature.com/articles/s41598-017-12059-2)；
[Nature Communications, 2014](https://www.nature.com/articles/ncomms5016)；
[Advanced Materials, 2012](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.201104403)）。

### 1.4 科学发现边界

需要区分以下四种陈述：

1. **模型描述**：模型对 EML 的注意力较高。
2. **数据关联**：当前数据中某类结构通常对应更高 EQE。
3. **物理假设**：该结构可能通过载流子平衡或激子限制提高 EQE。
4. **因果规律**：改变该结构会稳定地导致 EQE 提高。

本报告可以可靠支持第 1 层，部分支持第 2 层，可以生成第 3 层假设，但不能单独
完成第 4 层因果证明。

---

## 2. 当前 OLED-GAT 中的注意力到底是什么

### 2.1 图结构

当前图的层级为：

```text
device root
  -> ordered layer nodes
      -> material nodes
          -> atom/bond nodes for EML molecules
```

主要有以下有向边：

```text
root <-> layer
adjacent layer <-> layer
layer <-> material
EML material <-> atom
atom <-> atom bond
```

边类型作为可学习嵌入进入 GATv2 注意力计算。

### 2.2 GATv2 注意力

对于从源节点 `j` 指向目标节点 `i` 的消息，第 `h` 个注意力头产生系数：

```text
alpha(j -> i, h) = softmax over incoming neighbors of i
```

每个目标节点对所有入边的注意力为正且归一化。当前模型使用 GATv2，
因为它相较原始 GAT 具有动态、查询相关的邻居排序能力
（[GAT, ICLR 2018](https://openreview.net/forum?id=rJXMpikCZ)；
[GATv2, ICLR 2022](https://openreview.net/forum?id=F72ximsx7C1)）。

注意力系数回答的是：

> 当前消息传递模块在更新目标节点时，对不同来源消息分配了多少相对权重？

它不直接回答：

> 这条消息让最终 EQE 增加了多少？

原因包括：

- 注意力始终为非负值，不能直接表达提高或降低 EQE。
- 注意力在目标节点的邻居内部归一化，不同节点度数之间不能简单横向比较。
- 消息还经过线性变换、非线性、残差连接和 LayerNorm。
- 多个模块会反复混合信息。
- 最终读出不只依赖 device root。

### 2.3 当前读出结构对解释性的影响

模型最终拼接了五类图表示：

```text
device root
mean(all layer nodes)
mean(all material nodes)
mean(EML layer nodes)
mean(EML material nodes)
```

之后再与器件机理、颜色、工艺和器件类型等上下文拼接。

这意味着当前模型存在两条并行的信息路径：

1. GATv2 注意力控制的消息传递路径。
2. 对全部层和材料直接做均值池化的旁路。

因此，`layer -> root` 注意力只能解释第一条路径，无法完整解释某层通过均值池化
进入输出的影响。这是本次注意力忠实性只有中等水平的一个结构性原因。

---

## 3. 为什么不能把注意力直接当成解释

“Attention is not Explanation”表明，注意力可能与梯度重要性不一致，也可能存在
不同注意力分布却产生近似相同输出的情况
（[Jain and Wallace, NAACL 2019](https://aclanthology.org/N19-1357/)）。

对于多层注意力模型，信息会随着深度逐渐混合；直接观察最后一层注意力也不足以
恢复输入信息流。注意力 rollout 和 attention flow 等方法通常比单层原始权重更接近
消融和梯度结果
（[Abnar and Zuidema, ACL 2020](https://aclanthology.org/2020.acl-main.385/)）。

图神经网络解释研究也建议至少从以下维度验证解释：

- faithfulness：删除高重要性信息后输出是否显著变化；
- stability：轻微扰动或重新训练后解释是否稳定；
- consistency：不同方法是否得到一致结果；
- sparsity：解释是否足够集中；
- domain validity：解释是否满足领域知识。

相关方法包括
[GNNExplainer](https://proceedings.neurips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html)、
[Integrated Gradients](https://arxiv.org/abs/1703.01365) 和
[GNN attribution benchmark](https://proceedings.neurips.cc/paper_files/paper/2020/hash/417fbbf2e9d5a28a855a11894b2e795a-Abstract.html)。

因此，本项目采用联合证据，而不是只画注意力热图。

---

## 4. 本次实证分析方法

### 4.1 原始层注意力

对每个器件、每个 GATv2 模块和每个注意力头：

1. 找到所有 `layer -> device root` 边。
2. 排除 root self-loop。
3. 在器件内部重新归一化层注意力。
4. 得到每个功能层的信息路由份额。

该指标记为：

```text
AttentionShare(layer)
```

### 4.2 输出条件化注意力

为了让注意力与最终 EQE 输出关联，计算：

```text
|alpha * d(EQE prediction) / d(alpha)|
```

然后在同一器件、模块和注意力头的层边之间归一化。

该指标记为：

```text
AttentionGradientShare(layer)
```

它比原始注意力多回答了一个问题：

> 当前注意力系数的小变化是否会影响最终 EQE 输出？

但它仍然是局部一阶近似，而不是因果贡献。

### 4.3 厚度梯度

当前厚度输入使用：

```text
z = (log(1 + thickness_nm) - training_mean) / training_std
```

因此物理单位下的局部梯度为：

```text
d(EQE) / d(thickness_nm)
  = d(EQE) / dz / [training_std * (1 + thickness_nm)]
```

### 4.4 厚度反事实

对于处于训练集对应层角色第 1 至第 99 百分位范围内的厚度：

```text
f_low  = model(thickness * 0.9)
f_high = model(thickness * 1.1)

Delta_10% = (f_high - f_low) / 2
```

只改变目标层厚度，其余器件结构保持不变。

该值表示当前模型在该器件局部邻域内学到的厚度敏感性，不代表真实实验中改变厚度
后其他物理量保持不变。

### 4.5 根连接消融

删除目标层与 device root 之间的双向边，保持其余图结构不变：

```text
RootLinkEffect = |f(ablated graph) - f(original graph)|
```

用它检验高注意力层是否确实对应更大的模型输出变化。

注意：由于最终还有 layer mean pooling，该消融不是完整删除该层，只是在检验
注意力路径的忠实性。

### 4.6 稳定性

对每个模块计算：

- 4 个注意力头之间的层排序 Spearman 相关；
- 层注意力分布的归一化熵。

低相关表示不同头学习不同模式；低熵表示注意力更集中。

### 4.7 统计原则

同一论文往往包含多个相似器件，不能把所有器件当作完全独立样本。因此层角色汇总
采用：

1. 先在同一论文内求均值；
2. 再对论文层面的数值求中位数；
3. 以论文为聚类单位做 1,000 次 bootstrap 置信区间。

---

## 5. 实证结果

### 5.1 层角色注意力

下表中的注意力是“当器件包含该层时”的条件统计，不能理解为所有器件的无条件
平均贡献。

| 层角色 | 层数 | 论文数 | 注意力-梯度份额中位数 | 95% 聚类 bootstrap CI |
|---|---:|---:|---:|---:|
| EBL | 217 | 166 | 16.45% | 15.46%–16.99% |
| EML | 428 | 345 | 16.21% | 15.29%–17.64% |
| HTL | 399 | 300 | 14.99% | 14.38%–16.07% |
| EIL | 392 | 321 | 13.49% | 13.05%–13.81% |
| anode | 400 | 327 | 13.25% | 12.80%–13.79% |
| HIL | 301 | 244 | 9.48% | 8.85%–10.60% |
| ETL | 400 | 325 | 8.24% | 7.60%–9.30% |
| cathode | 400 | 328 | 8.08% | 7.67%–8.43% |
| HBL | 107 | 91 | 6.38% | 5.33%–9.06% |

生成图路径：`outputs_device_random/explainability/layer_role_attention_and_thickness.png`。
模型输出未包含在公开代码仓库中，可通过 `run_explainability.py` 重新生成。

### 5.2 结果解释

#### EML

EML 的高注意力符合当前模型输入结构：

- EML 材料具有 SMILES 分子图；
- EML 单独参与一个专门的池化分支；
- 发光机理、客体材料及掺杂信息集中在 EML；
- EQE 的激子形成、辐射跃迁和浓度猝灭主要发生于 EML。

因此 EML 的高注意力既包含真实物理信息，也包含模型架构预先赋予 EML 的特权。

#### EBL

EBL 在存在时获得最高的条件注意力。可能原因包括：

- EBL 是可选层，其出现本身具有较高辨识度；
- 它直接位于 EML 空穴侧，可改变电子泄漏、激子限制和复合区位置；
- 含 EBL 的器件可能集中在特定高性能器件家族。

不能据此得出“增加 EBL 必然提高 EQE”，因为 EBL 的存在与材料体系、年代和论文
设计策略存在混杂。

#### HTL、EIL 与电极

HTL、EIL 和 anode 的注意力较高，说明模型大量使用注入侧结构。它可能捕捉：

- 载流子注入难度；
- 器件是常规还是倒置结构；
- 常用材料组合与论文家族；
- 电极和注入层的类别模式。

当前模型没有显式 HOMO、LUMO、功函数和迁移率，因此不能进一步分辨模型到底使用
的是能级对齐、材料身份记忆还是器件类型代理变量。

### 5.3 不同消息传递深度

生成图路径：`outputs_device_random/explainability/attention_gradient_by_role_and_block.png`。

观察到明显的深度分工：

- Block 1 更分散，anode、EIL、cathode 等基础结构获得较多注意力。
- Block 2 对 anode、EIL 及部分 EML 邻域进一步聚合。
- Block 3 开始明显集中于 EML 和 HTL/EBL。
- Block 4 对 EML、EBL、HTL 和 HBL 更集中。

这与“浅层编码局部类别和边类型，深层形成器件级 EML 邻域表示”的解释相容。

### 5.4 层界面消息

生成图路径：`outputs_device_random/explainability/interface_attention_by_block.png`。

第 4 个模块中最强的有向界面消息为：

| 消息方向 | 平均绝对 attention-gradient，EQE 单位 |
|---|---:|
| HTL -> EML | 0.0723 |
| EBL -> EML | 0.0651 |
| HIL -> EML | 0.0545 |
| HBL -> EML | 0.0315 |
| ETL -> EML | 0.0223 |

这说明深层模型主要在构造一个由 EML 两侧输运和阻挡层共同决定的 EML 表示。

从 OLED 物理上，这一模式可能对应：

- 空穴侧注入和空穴输运；
- 电子侧电子输运；
- 电子、空穴及激子的限制；
- 复合区在 EML 内的位置；
- EML 两侧界面猝灭。

但是，当前边上只有层顺序、层角色和材料表示，没有显式能级差。因此该图只能表明
模型使用了“哪个界面”，不能说明模型识别了“多大的势垒”或“哪一种泄漏机制”。

### 5.5 注意力忠实性

生成图路径：`outputs_device_random/explainability/attention_faithfulness.png`。

| 指标 | 原始注意力 | Attention × Gradient |
|---|---:|---:|
| 全部层 pooled Spearman | 0.503 | 0.480 |
| 器件内部 Spearman 中位数 | 0.536 | 0.533 |
| 器件内部 Q25–Q75 | 0.286–0.714 | 0.305–0.679 |

解释：

- 相关性为正，说明注意力不是完全无意义。
- 相关性仅为中等，说明高注意力不总是对应大消融效应。
- Attention × Gradient 没有明显优于原始注意力。
- 均值池化旁路、残差连接、多跳消息和非线性共同削弱了一一对应关系。

因此，模型 UI 中可以显示注意力，但必须标记为：

```text
Model attention / information routing
```

而不能标记为：

```text
Physical contribution to EQE
```

### 5.6 注意力头稳定性

| GATv2 模块 | 头间排序相关中位数 | 注意力归一化熵中位数 |
|---|---:|---:|
| Block 1 | 0.053 | 0.974 |
| Block 2 | 0.133 | 0.702 |
| Block 3 | 0.524 | 0.665 |
| Block 4 | -0.083 | 0.559 |

主要发现：

- Block 1 接近均匀注意力，各头排序几乎不一致。
- Block 3 的头之间最一致。
- Block 4 更集中，但不同头排序呈轻微负相关，表明最终模块的头可能分别关注不同
  器件路径。

因此不应只展示四头平均图。更合理的前端展示是：

```text
Consensus attention
Head-specific views
Head disagreement warning
```

还需要通过至少 5 个随机种子重新训练，检验这种头分工是否跨模型稳定。

---

## 6. 层厚度对预测 EQE 的影响

### 6.1 全局结果

| 层角色 | 可分析层数 | \|上下 10% 厚度导致的 EQE 变化\|中位数 | 有符号变化中位数 |
|---|---:|---:|---:|
| EML | 364 | 0.0174 | -0.0174 |
| EBL | 136 | 0.00290 | +0.00042 |
| HTL | 366 | 0.00245 | +0.00129 |
| HIL | 275 | 0.00191 | +0.00153 |
| anode | 104 | 0.00176 | +0.00176 |
| HBL | 63 | 0.00174 | +0.00170 |
| ETL | 359 | 0.00124 | -0.00049 |
| cathode | 287 | 0.00045 | +0.00041 |
| EIL | 343 | 0.00041 | +0.00039 |

这里的单位是 **EQE 百分点**。例如 EML 的 0.0174 不是 1.74%，而是预测从
20.000% 变为约 19.983% 的量级。

生成图路径：`outputs_device_random/explainability/signed_thickness_effect_by_role.png`。

### 6.2 EML 的局部负相关

在 364 个可分析 EML 中：

- 85.7% 的器件局部斜率为负；
- 有符号效应中位数为 -0.0163 个 EQE 百分点；
- 第 10 至第 90 百分位约为 -0.0388 至 +0.00162。

模型学到的统计模式是：

> 在当前数据分布和其他输入固定时，增厚 EML 通常轻微降低预测 EQE。

不能直接将其解释为实验规律，原因包括：

- 高 EQE 器件可能更常采用较薄 EML，形成选择偏差。
- 同论文不同器件的厚度、掺杂浓度和材料可能同时变化。
- 模型没有光学常数、复合区位置和载流子迁移率。
- 10% 的单变量修改不保证器件仍满足电学和光学自洽。
- 效应量相对于模型 MAE 3.45 个百分点极小。

### 6.3 HBL 的局部正相关

在 63 个可分析 HBL 中，约 90.5% 的局部厚度效应为正，但中位数只有
+0.00169 个 EQE 百分点。

它最多只能形成以下弱假设：

> 在当前训练分布内，模型倾向于将稍厚 HBL 与略高 EQE 关联。

它不能支持：

> 增厚 HBL 会显著提高 EQE。

### 6.4 数值正确性检查

局部解析梯度与上下 10% 有限差分高度一致：

```text
Spearman rho = 0.99976
median absolute discrepancy = 1.28e-5 EQE percentage points
```

这说明厚度分析代码数值上可靠。效应小不是计算错误，而是冻结模型确实没有强烈
使用厚度数值。

---

## 7. 从可解释性反向诊断当前模型

### 7.1 厚度特征被弱使用

当前模型虽然输入厚度，但与材料身份、层角色和 EML 分子指纹相比，厚度信息对输出
影响很小。

可能原因：

- 厚度只占 layer numeric vector 的一个连续维度。
- 材料和器件家族类别具有更强预测信号。
- 不同材料体系的最优厚度不同，混在一起会抵消全局规律。
- 厚度缺失率和自动抽取噪声降低了其可信度。
- 当前 device-random split 允许同论文器件跨集合，模型更容易利用器件家族信息。

### 7.2 第 4 个原子更新模块存在“输出不可达”消息

解释梯度显示，第 4 个模块中以下目标为 atom 的边对最终输出梯度为零：

```text
material -> atom
atom -> atom bond
```

原因是第 4 个模块完成后，最终读出没有对 atom 节点池化。最后一次更新得到的 atom
表示不会再传回 material，因而不可能影响输出。

这不是预测错误，但说明：

- 最后一层部分原子消息计算是冗余的。
- 当前同质全图四层 GAT 不够符合层级信息流。
- 更合理的结构应先独立完成分子编码，再把分子表示送入层级器件图。

推荐结构：

```text
Stage A: molecular atom/bond GNN
         -> material embedding

Stage B: material-to-layer attention
         -> layer embedding

Stage C: directed adjacent-layer/interface attention
         -> device embedding

Stage D: mean + quantile EQE heads
```

这种分层模型既减少无效计算，也让“分子、层、界面、器件”四级解释更清晰。

### 7.3 当前注意力无法解释最终集成预测

当前最佳结果：

```text
final prediction = 0.62 * CatBoost + 0.38 * OLED-GAT
```

所以完整解释应为：

```text
Final attribution
  = 0.62 * CatBoost attribution
  + 0.38 * OLED-GAT attribution
```

CatBoost 分量需要 SHAP 或相同的输入反事实分析。只展示 GAT 注意力会遗漏最终预测
中的主要分量。

### 7.4 当前解释只针对均值头

本次解释目标是 mean EQE head。q10、q50 和 q90 可能依赖不同层信息。

未来应分别分析：

- 哪些层推动预测均值；
- 哪些层扩大 q90-q10 区间；
- 哪些输入导致模型不确定。

这会使解释从“为什么预测高 EQE”扩展到“为什么这个预测不可靠”。

---

## 8. 如何把注意力升级为层参数科学解释

## 8.1 第一优先级：增加真正的物理层参数

当前需要增加的关键字段：

### 每个材料

```text
HOMO
LUMO
triplet energy T1
singlet energy S1
hole mobility
electron mobility
PLQY
refractive index / extinction coefficient
dipole orientation
```

### 每个相邻层界面

```text
hole injection barrier
electron injection barrier
HOMO offset
LUMO offset
triplet confinement barrier
mobility mismatch
interface polarity
```

### 每个 EML

```text
host/guest ratio
sensitizer/final-emitter ratio
EML PLQY
emission mechanism
horizontal dipole ratio
```

这些变量加入后，注意力才能从“HTL -> EML 很重要”升级为：

> 模型认为 HTL/EML 的 HOMO 势垒和空穴迁移率失配控制了当前器件的电荷平衡。

### 8.2 第二优先级：把界面变成显式建模对象

目前界面只是两个 layer node 之间的一条边。推荐为每个相邻界面构造显式特征：

```text
interface_feature = [
  source_role,
  target_role,
  delta_HOMO,
  delta_LUMO,
  delta_T1,
  thickness_left,
  thickness_right,
  log_mobility_ratio,
]
```

然后输出：

```text
interface attention
interface gradient
interface counterfactual effect
```

这种解释比单独给某层分配注意力更符合 OLED 物理，因为注入、阻挡和泄漏本质上是
界面现象。

### 8.3 第三优先级：加入输出专用的层注意力读出

推荐在现有均值池化之外增加：

```text
beta_i = softmax(
  MLP(layer_i, device_root, context)
)

device_layer_attention = sum_i beta_i * layer_i
```

需要保留 mean pooling 作为稳健旁路，不能只用 attention pooling。

对不同输出使用不同注意力：

```text
beta_mean
beta_q10
beta_q50
beta_q90
```

这样可以区分性能驱动层和不确定性驱动层。

### 8.4 第四优先级：解释方法集成

每个器件至少同时输出：

1. 原始 attention。
2. Attention × Gradient。
3. Integrated Gradients。
4. 层角色或材料身份遮蔽。
5. 厚度/掺杂比例反事实曲线。
6. GNNExplainer 或 PGExplainer 子图。

只有多个方法方向一致时，才将结论升级为“高可信模型解释”。

### 8.5 第五优先级：同论文匹配器件分析

当前数据库很适合构建比普通全局相关更强的准因果证据。

选择同一篇论文中满足以下条件的器件对：

```text
same emitter
same host
same major stack
same fabrication method
only one layer/material/thickness differs
```

计算：

```text
observed Delta EQE
predicted Delta EQE
attention shift
counterfactual Delta EQE
```

同论文匹配可以部分控制：

- 实验室和测量条件；
- 材料批次；
- 器件制备平台；
- 文献年代；
- 报告习惯。

这是从“模型解释”走向“器件规律”的关键步骤。

---

## 9. 面向 4CzIPN/PPF 实验故事的应用

用户的实验故事是：

```text
mCP / 4CzIPN EML
-> TmPyPB HOMO 较浅，存在空穴泄漏
-> 引入深 HOMO 的 PPF 作为 HBL
-> 提高激子限制和电荷平衡
-> EQE 从 18.6% 提升到 26.5%
```

当前注意力结果能够提供以下总体支持：

- 模型深层确实关注指向 EML 的 HTL/EBL/HBL/ETL 界面消息。
- HBL->EML 在第 4 模块中成为高重要性界面之一。
- OLED 数据集整体支持“EML 边界联合决定器件表现”这一模型假设。

但当前模型不能独立重建 PPF 的物理机制，原因是：

- 没有 PPF 与 mCP、TmPyPB 的 HOMO/LUMO 数值。
- 没有空穴泄漏势垒。
- 没有迁移率和复合区信息。
- 模型对 4CzIPN/PPF 示例本身预测偏低。

为了让该故事成为可发表的智能体发现链条，建议建立如下证据：

```text
Evidence A: global attention
EML boundary interfaces are repeatedly important across the corpus.

Evidence B: physically parameterized model
Delta HOMO and Delta LUMO explain interface attention and EQE.

Evidence C: matched literature devices
Deep-HOMO HBL variants outperform matched controls.

Evidence D: local counterfactual
Replacing TmPyPB-adjacent interface with PPF predicts higher EQE.

Evidence E: experiment
Observed EQE improves from 18.6% to 26.5%.
```

其中 A 已有初步结果，B–D 仍需开发，E 已由实验提供。

---

## 10. 推荐的实施路线

### Phase 1：当前模型解释功能产品化

目标：把本报告中的离线解释用于单器件分析。

- 导出每篇器件的层注意力和 attention-gradient。
- 在器件堆叠图上显示层颜色强度。
- 点击某层显示四个注意力头。
- 提供厚度滑块，实时显示预测曲线。
- 明确标记“model association, not causal effect”。

### Phase 2：解释最终集成模型

- OLED-GAT 使用本报告方法。
- CatBoost 使用 SHAP 和输入反事实。
- 按 0.38/0.62 权重合并到最终预测解释。
- 对 GAT 与 CatBoost 解释冲突的器件做风险标记。

### Phase 3：能级与界面特征

- 开发 HOMO/LUMO/T1 数据挖掘模块。
- 构建 material energy-level provenance。
- 计算相邻层能级差。
- 将 interface feature 加入图。
- 重新做 DOI-grouped 训练与解释。

### Phase 4：解释忠实性验证

- 至少 5 个随机种子。
- 计算跨种子层排序稳定性。
- 比较 attention、IG、GNNExplainer 和遮蔽。
- 采用 top-k deletion、bottom-k deletion、sufficiency 和 necessity 指标。
- 在人工构造的物理对照器件上验证解释。

### Phase 5：器件规律与实验验证

- 建立同论文匹配器件对。
- 分机制、颜色、工艺分别估计层参数效应。
- 只将跨论文、跨种子、跨解释方法稳定的规律提交实验。
- 用实验观察更新模型和解释可信度。

---

## 11. 可发表结论与不可发表结论

### 当前可以谨慎陈述

> 在当前 OLED-GAT 中，输出条件化注意力主要集中于 EML、EBL 和 HTL，
> 深层消息传递优先聚合指向 EML 的输运与阻挡层界面信息。这说明模型预测
> EQE 时主要依赖发光层及其边界结构。

> 注意力与根连接消融具有中等相关性，说明注意力可用于定位模型信息路由，
> 但不足以作为独立的物理归因。

> 当前模型对厚度数值的敏感性远低于对层角色和材料信息的敏感性，表明模型尚未
> 学到可用于器件厚度优化的强定量规律。

### 当前不能陈述

```text
EBL 对 EQE 的贡献是 16.45%。
```

错误原因：16.45% 是条件化、归一化的模型注意力份额，不是物理贡献率。

```text
EML 增厚一定降低 EQE。
```

错误原因：只是当前模型中的微弱局部关联，且存在严重混杂。

```text
HBL 增厚会提高 EQE。
```

错误原因：效应量极小，样本较少，且未控制材料和能级。

```text
该注意力图解释了 R2=0.753 的最终模型。
```

错误原因：最终模型 62% 来自 CatBoost。

---

## 12. 最终判断

从注意力机制出发研究 OLED-GAT 可解释性是可行的，但需要明确其功能定位。

### 注意力最适合做什么

- 定位模型关注的功能层。
- 定位重要的相邻层界面。
- 发现模型是否依赖非预期特征。
- 为同论文对照分析和实验设计生成假设。
- 在前端为研究者提供器件级解释入口。

### 注意力不能单独做什么

- 给出层对 EQE 的物理贡献百分比。
- 判断参数改变的正负方向。
- 证明载流子或激子机制。
- 证明某种层设计具有因果增益。

### 当前最重要的研究结论

> 当前 OLED-GAT 已经学到“EML 及其两侧界面是器件 EQE 预测核心”的结构性模式，
> 但尚未学到足够强、足够物理化的层参数规律。下一步不应继续美化注意力图，
> 而应加入能级、迁移率、PLQY、取向和界面势垒，并以同论文匹配器件和实验
> 反事实验证注意力所提出的假设。

---

## 13. 复现方式

完整分析：

```bash
CUDA_VISIBLE_DEVICES=0 uv run python \
  analysis/oled_gat/run_explainability.py
```

只重新生成图：

```bash
uv run python \
  analysis/oled_gat/run_explainability.py \
  --plots-only
```

主要输出目录：

```text
analysis/oled_gat/outputs_device_random/explainability/
```

关键文件：

```text
layer_attributions.csv
layer_role_summary.csv
edge_attributions.csv
edge_type_summary.csv
interface_attributions.csv
interface_summary.csv
attention_head_stability.csv
explainability_summary.json
attention_gradient_by_role_and_block.png
layer_role_attention_and_thickness.png
attention_faithfulness.png
signed_thickness_effect_by_role.png
interface_attention_by_block.png
```

## 14. 参考文献

1. Veličković P. et al. Graph Attention Networks.
   [ICLR 2018](https://openreview.net/forum?id=rJXMpikCZ).
2. Brody S., Alon U., Yahav E. How Attentive are Graph Attention Networks?
   [ICLR 2022](https://openreview.net/forum?id=F72ximsx7C1).
3. Jain S., Wallace B. Attention is not Explanation.
   [NAACL 2019](https://aclanthology.org/N19-1357/).
4. Abnar S., Zuidema W. Quantifying Attention Flow in Transformers.
   [ACL 2020](https://aclanthology.org/2020.acl-main.385/).
5. Ying Z. et al. GNNExplainer: Generating Explanations for Graph Neural Networks.
   [NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/d80b7040b773199015de6d3b4293c8ff-Abstract.html).
6. Sundararajan M., Taly A., Yan Q. Axiomatic Attribution for Deep Networks.
   [ICML 2017](https://arxiv.org/abs/1703.01365).
7. Sanchez-Lengeling B. et al. Evaluating Attribution for Graph Neural Networks.
   [NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/hash/417fbbf2e9d5a28a855a11894b2e795a-Abstract.html).
8. Amara K. et al. GraphFramEx: Towards Systematic Evaluation of Explainability
   Methods for Graph Neural Networks.
   [arXiv 2022](https://arxiv.org/abs/2206.09677).
9. Lee J. et al. The Role of Charge Balance and Excited State Levels on Device
   Performance of Exciplex-based Phosphorescent OLEDs.
   [Scientific Reports 2017](https://www.nature.com/articles/s41598-017-12059-2).
10. Uoyama H. et al. High-efficiency organic light-emitting diodes with fluorescent
    emitters.
    [Nature Communications 2014](https://www.nature.com/articles/ncomms5016).
11. Pu Y.-J. et al. Optimizing the Charge Balance of Fluorescent OLEDs to Achieve
    High EQE Beyond the Conventional Upper Limit.
    [Advanced Materials 2012](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adma.201104403).
12. Brütting W. et al. Device efficiency of OLEDs: Progress by improved light
    outcoupling.
    [physica status solidi 2013](https://onlinelibrary.wiley.com/doi/abs/10.1002/pssa.201228320).
