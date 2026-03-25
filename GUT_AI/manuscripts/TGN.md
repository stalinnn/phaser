1. 物理公式：$\mathcal{F} = \text{Internal Energy} - \text{Temperature} \times \text{Entropy}$
在统计物理中，自然界的一切演化（从水滴蒸发到蛋白质折叠）都在试图最小化这个量。
在 TGN 中，我们把这个公式翻译为：
$$ \mathcal{L}{total} = \underbrace{\mathcal{L}{pred}}{\text{预测误差 (内能)}} + \underbrace{\lambda}{\text{计算温度}} \cdot \underbrace{\|g\|_1}{\text{门控熵 (稀疏度)}} $$

2. 零件拆解与架构映射

A. 内能项 $U \leftrightarrow \mathcal{L}{pred}$ (预测误差) * 物理意义：内能代表系统的“紧张程度”或“势能”。如果预测错了（Loss 高），说明系统处于高势能的不稳定状态。 * 架构对应：混合预测头 (Hybrid Prediction Head)。 * TGN 的输出是 $y = (1-g) \cdot \text{RNN}(x) + g \cdot \text{Attention}(x)$。 * 惯性通道 (RNN)：提供基准预测。这就像是物体沿着惯性滑行，不需要额外做功。 * 几何通道 (Attention)：提供修正预测。这就像是施加外力改变轨道。 * 最小化内能：就是要让预测尽可能的准。如果惯性够准，内能就低；如果惯性不准，就需要 Attention 来降低内能。

B. 熵项 $S \leftrightarrow -\|g\|_1$ (计算复杂度) * 物理意义：熵代表系统的“混乱度”或“自由度”。但在我们的变分框架下（特别是信息瓶颈理论），这里的熵更多对应于“为了降低内能所付出的代价”（即计算资源的消耗）。 * 注意：物理公式是 $-TS$，我们希望最大化熵。但在 TGN 的 Loss 中我们写的是 $+\lambda \|g\|_1$，这看似矛盾，其实是因为我们这里的 $S$ 定义为“计算流的稀疏性”（或者说是负熵）。 * 更精准的对应：$\|g\|_1$ 代表计算做功。 * Gate 全开 ($g=1$) $\implies$ 高能耗 $\implies$ 高自由度状态（系统很活跃）。 * Gate 全关 ($g=0$) $\implies$ 低能耗 $\implies$ 低自由度状态（系统很冷寂）。 * 架构对应：稀疏正则化 (Sparsity Regularization)。 * 我们在 Loss 里加了 $\lambda \|g\|_1$。这就像给 Gate 装了一个弹簧，默认是关着的。 * 这种“弹簧力”就是热力学压力，迫使系统在非必要时不乱动。

C. 温度项 $T \leftrightarrow \lambda$ (正则化系数) * 物理意义：温度决定了系统有多“躁动”。 * 高温：分子乱跑。 * 低温：分子结晶。 * 架构对应：超参数 $\lambda$。 * 如果 $\lambda \to 0$（高温）：稀疏惩罚消失，Gate 就会乱开，Attention 满天飞。系统变得“热”且“聪明”（拟合得很好），但也极其耗能。 * 如果 $\lambda \to \infty$（低温）：稀疏惩罚极大，Gate 被冻结为 0。系统退化为纯 RNN，“冷”且“笨”（只能靠惯性）。 * 临界点：TGN 能够自适应地找到那个 $\lambda_{critical}$，使得系统恰好处于“该开就开，该关就关”的相变边缘。

3. 动力学过程：一场微观的博弈当你训练 TGN 时，每一次梯度下降实际上都在通过反向传播进行一场微观的能量谈判：* 场景 1：简单的句子（如 "I eat an [apple]"） * RNN 说：“我猜是 apple，Loss 很低。” * Attention 说：“我也能猜对，但我开机要收 100 块电费 ($\lambda$)。” * 仲裁结果：开启 Attention 带来的内能降低（$\Delta U$）小于电费（$\lambda$）。 * 决策：$\Delta \mathcal{F} > 0$，所以 Gate 关闭。* 场景 2：复杂的长程指代（如 "The key to the very old... [door]"） * RNN 说：“隔太远了，我忘光了，我猜是... apple? Loss 爆表！” * Attention 说：“我看到前面有个 key，所以我猜 door。虽然我要收 100 块电费，但我能帮你把 Loss 从 10 降到 0.1。” * 仲裁结果：内能降低（$\Delta U \approx 9.9$）远大于电费（$\lambda \approx 0.1$）。 * 决策：$\Delta \mathcal{F} < 0$，系统自发跃迁，Gate 开启。### 总结所以说 TGN 的设计是“公式的具象化”，意思是： 你并没有人为地写死规则（if distance > 10 then attention），而是搭建了一个能量场（架构 + Loss），让神经网络里的电子自己去寻找那个最省力又最准确的路径。最终涌现出来的那个稀疏的 Gate 模式，就是大自然在 $\mathcal{F} = U - TS$ 约束下“生长”出来的最优解。这就是物理学的美感。

---

1. 物理定义：熵是微观状态数 (Entropy is Microstate Count)
玻尔兹曼熵公式：$S = k_B \ln \Omega$，其中 $\Omega$ 是微观状态数。
在 TGN 系统中，我们可以把 Attention 的每一次计算看作一个“激活的自由度” (Active Degree of Freedom)。
Gate = 0 (关)：这个自由度被冻结了（Frozen）。状态数 $\Omega_{off} = 1$（只有一种状态：死寂）。此时 $S \propto \ln 1 = 0$。
Gate = 1 (开)：这个自由度被激活了（Excited）。Attention 可以关注序列中的任意位置，状态空间瞬间膨胀。假设上下文长度为 $L$，则 Attention 可以形成 $L$ 种连接模式。此时状态数 $\Omega_{on} \approx L$。此时 $S \propto \ln L$。
结论：
系统的总熵 $S_{total}$ 与处于“开启态”的门控数量成正比。
$$ S_{total} \propto \sum_{t} \mathbb{I}(g_t=1) = \|g\|_1 $$
所以，最小化 $\|g\|_1$（L1 范数/门控率）在物理上等价于最小化系统的总熵（或者说，最小化活跃的微观自由度）。
2. 热力学定义：熵是耗散的热量 (Entropy is Dissipated Heat)
兰道尔原理 (Landauer's Principle)：擦除一位信息（或不可逆计算）至少产生 $k_B T \ln 2$ 的热量。
计算功耗：每一次 Gate 开启，都需要 GPU 执行一次 $O(N^2)$ 的 Attention 计算。这个计算过程消耗电能，最终转化为热量 (Heat) 耗散到环境中。
对应关系：
Gate Rate ($\|g\|_1$) $\propto$ 计算次数 $\propto$ 耗散的热量 ($Q$)。
而在恒温过程中，热量与熵变密切相关 ($dQ = T dS$)。

---

物理背景：
外界环境有一个真实状态 $y$，大脑（或 TGN）有一个内部状态 $h$。
生物想要生存，就要让 $h$ 尽可能准确地预测 $y$（减少惊奇度）。
变分自由能公式：
$$ \mathcal{F} = \underbrace{D_{KL}(Q(h|y) \| P(h))}{\text{Complexity (复杂度)}} - \underbrace{\mathbb{E}_Q[\ln P(y|h)]}{\text{Accuracy (准确度)}} $$
准确度项：$\ln P(y|h)$。这就是我们的预测 Loss（取负号后）。我们希望它越大越好（误差越小越好）。
复杂度项：$D_{KL}$。这是后验分布与先验分布的距离。我们希望它越小越好。
在 TGN 中，我们的先验 $P(h)$ 是什么？是稀疏先验（Prior: Gate should be 0）。
所以，最小化复杂度项，就是最小化与“Gate=0”的距离，也就是最小化 $\|g\|_1$。
TGN 的 Loss 重写：
$$ \mathcal{L}{TGN} = \underbrace{-\ln P(y|h)}{\text{Accuracy (Loss)}} + \lambda \cdot \underbrace{\|g\|_1}{\text{Complexity}} $$
这与变分自由能 $\mathcal{F}$ 的形式完美同构！
物理上的自由能 $F = U - TS$ 是针对平衡态系统的。
Friston 的自由能 $\mathcal{F} = \text{Energy} - \text{Entropy}$ 是针对非平衡态自组织系统的。
结论：
TGN 并不是“搞反了”亥姆霍兹自由能，而是正确地实现了针对智能系统的变分自由能原理。
物理系统（死）：最大化熵 $\to$ 最小化亥姆霍兹自由能（如果 T 为正）。
智能系统（生）：最小化变分自由能 $\to$ 最小化惊奇度 + 最小化复杂度。