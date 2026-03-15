## 1. 权重初始化

### 1.1 为什么要关心初始化？

神经网络的训练本质是优化损失函数，通过梯度下降更新权重。如果初始化不当：

- **全零初始化**：所有神经元输出相同，梯度相同，无法打破对称性，网络无法学习。
- **初始值过大**：激活函数（如 sigmoid）进入饱和区，梯度接近零，梯度消失。
- **初始值过小**：信号逐层衰减，深层网络难以训练。

合适的初始化能**保持前向/反向传播中方差稳定**，避免梯度消失/爆炸，加速收敛。

### 1.2 常见初始化方法及原理

#### (1) 随机初始化（简单高斯/均匀分布）
直接从高斯分布 \( \mathcal{N}(0, \sigma^2) \) 或均匀分布 \( U(-r, r) \) 采样。但 \(\sigma\) 或 \(r\) 需要精心选择，否则方差失控。

#### (2) Xavier/Glorot 初始化
- **原理**：假设激活函数为线性（如 tanh 的近似线性区），希望每层输出的方差等于输入的方差，且反向传播时梯度的方差也保持一致。  
  推导得到：对于第 \(l\) 层，输入维度 \(fan\_in\)，输出维度 \(fan\_out\)，则  
  \[
  Var(W) = \frac{2}{fan\_in + fan\_out}
  \]
  通常实现为均匀分布 \( U\left(-\sqrt{\frac{6}{fan\_in + fan\_out}}, \sqrt{\frac{6}{fan\_in + fan\_out}}\right) \)（Xavier uniform）或正态分布 \( \mathcal{N}\left(0, \sqrt{\frac{2}{fan\_in + fan\_out}}\right) \)。

- **适用**：tanh、sigmoid 等关于原点对称且在线性区间的激活函数。

- **通俗类比**：想象一个水管网络，Xavier 确保流入每个节点的水量与流出水量大致相等，不会因为管道粗细变化导致某处水压暴增或骤降。

#### (3) Kaiming/He 初始化
- **原理**：针对 ReLU 及其变体（PReLU、Leaky ReLU）设计。ReLU 会将一半神经元置零，导致方差减半。因此需放大初始权重方差来补偿。  
  对于 ReLU，建议  
  \[
  Var(W) = \frac{2}{fan\_in}
  \]
  实现为均匀分布 \( U\left(-\sqrt{\frac{6}{fan\_in}}, \sqrt{\frac{6}{fan\_in}}\right) \)（He uniform）或正态分布 \( \mathcal{N}\left(0, \sqrt{\frac{2}{fan\_in}}\right) \)。

- **适用**：ReLU、Leaky ReLU、PReLU 等。

- **通俗类比**：如果水龙头（ReLU）经常关掉一半（负值置零），为了保证下游水量不减少，初始时得把水压调大一些。

### 1.3 PyTorch 中的默认初始化

PyTorch 中常见的 `nn.Linear` 和 `nn.Conv2d` 默认采用 **Kaiming 均匀初始化**（针对 ReLU 类激活）。具体来说：

- 权重：`torch.nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))`  
  这里的 `a` 是激活函数的负斜率（用于 Leaky ReLU），默认 `sqrt(5)` 是为配合偏置初始化，实际对普通 ReLU 也可用。若想严格遵循 He 原论文，应设 `a=0`（即普通 ReLU）。
- 偏置：若存在，初始化为均匀分布 \( U(-\frac{1}{\sqrt{fan\_in}}, \frac{1}{\sqrt{fan\_in}}) \)。

**注意**：不同版本的 PyTorch 可能有细微差异，但总体遵循 Kaiming 初始化。如果使用 tanh，建议手动改为 Xavier。

### 1.4 PyTorch 代码示例

```python
import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # 选择初始化方式
        # 1. Xavier 均匀初始化 (适用于 tanh)
        # nn.init.xavier_uniform_(m.weight)
        # 2. Kaiming 均匀初始化 (适用于 ReLU)
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# 构建一个小模型
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 应用初始化
model.apply(init_weights)

# 验证第一层权重的统计量
print(model[0].weight.mean().item(), model[0].weight.std().item())
```

---

## 2. 归一化层

归一化层通过调整中间特征的分布，加速训练、提高稳定性。下面重点对比 **BatchNorm、LayerNorm、InstanceNorm、GroupNorm**。

### 2.1 为什么需要归一化？

- **内部协变量偏移**：网络参数更新导致各层输入分布变化，迫使后续层不断适应，降低训练速度。
- 归一化将每层输入拉回到均值为 0、方差为 1 的分布，允许使用更大的学习率，减少对初始化的依赖，并有一定正则化效果。

### 2.2 四大归一化对比

| 类型               | 归一化维度                               | 计算公式（简化）                                                                            | 适用场景                 | 通俗类比                 |
| ---------------- | ----------------------------------- | ----------------------------------------------------------------------------------- | -------------------- | -------------------- |
| **BatchNorm**    | 对 **Batch** 和 **空间** 维度做统计，对每个通道独立做 | \( \frac{x - \mu_{\text{batch}}}{\sigma_{\text{batch}}} \cdot \gamma + \beta \)     | CNN（依赖较大 batch）      | 全班同学的成绩按科目标准化，看相对位置  |
| **LayerNorm**    | 对 **特征** 维度（C, H, W）做统计，每个样本独立      | \( \frac{x - \mu_{\text{sample}}}{\sigma_{\text{sample}}} \cdot \gamma + \beta \)   | NLP（Transformer）、RNN | 每个学生自己的各科成绩标准化，关注自身  |
| **InstanceNorm** | 对每个样本的每个通道单独统计（H, W）                | \( \frac{x - \mu_{\text{channel}}}{\sigma_{\text{channel}}} \cdot \gamma + \beta \) | 风格迁移、图像生成            | 每张照片单独调亮度，不考虑和其他照片比较 |
| **GroupNorm**    | 将通道分组，每组内做归一化（G, H, W）              | 类似 LayerNorm，但只在组内统计                                                                | 小 batch 的视觉任务        | 将科目分组（文/理），组内标准化     |

#### 详细说明：

- **BatchNorm**：训练时使用当前 batch 的均值和方差；测试时使用全局移动平均统计量。因此 batch size 不宜太小（<8），否则统计不稳定。
- **LayerNorm**：不依赖 batch，每个样本自己算均值和方差，适合变长序列（NLP）。
- **InstanceNorm**：常用于图像风格迁移，因为要保留每个实例的独特风格。
- **GroupNorm**：GN 将通道分成若干组，每组内做归一化，是 LN 和 IN 的折中，当 batch size 很小时（如目标检测）效果优于 BN。

### 2.3 PyTorch 代码示例

```python
import torch
import torch.nn as nn

# 假设输入形状 (N, C, H, W) = (8, 64, 32, 32)
x = torch.randn(8, 64, 32, 32)

# BatchNorm (对每个通道)
bn = nn.BatchNorm2d(64)          # 参数 num_features = C
y_bn = bn(x)                      # 训练模式默认使用当前batch统计，包含可学习参数γ,β

# LayerNorm (通常对NLP，对图像可对整个(C,H,W)或最后一维)
ln = nn.LayerNorm([64, 32, 32])   # 需指定归一化维度大小
y_ln = ln(x)

# InstanceNorm (对每个样本每个通道)
inn = nn.InstanceNorm2d(64)       # 默认不包含可学习参数，若需设置affine=True
y_inn = inn(x)

# GroupNorm (将64通道分成8组，每组8通道)
gn = nn.GroupNorm(8, 64)          # num_groups=8, num_channels=64
y_gn = gn(x)

print(f"BN out shape: {y_bn.shape}, mean={y_bn.mean():.2f}, std={y_bn.std():.2f}")
```

**注意事项**：
- BN 在 `model.eval()` 时会切换到使用累积的全局统计量，务必正确设置训练/测试模式。
- LN 和 GN 在训练和测试时行为一致，无需特殊处理。

---

## 3. 残差连接

### 3.1 背景：深层网络的退化问题

直觉上，网络越深表达能力越强，但实际中过深的普通网络（如 VGG 堆叠）会出现**退化现象**：训练误差和测试误差反而升高。这不是过拟合，而是优化困难，深层网络难以学习恒等映射。

### 3.2 残差思想

**核心**：与其让网络直接学习期望的映射 \( \mathcal{H}(x) \)，不如学习残差 \( \mathcal{F}(x) = \mathcal{H}(x) - x \)，然后通过跳跃连接（shortcut）相加：  
\[
\mathcal{H}(x) = \mathcal{F}(x) + x
\]
如果恒等映射是最优的，网络只需将残差部分逼近零，比直接拟合恒等映射更容易。

**梯度优势**：反向传播时，梯度可以沿着跳跃连接直接回传，避免了中间层梯度的连乘衰减，缓解梯度消失。

### 3.3 ResNet 基础块（以 BasicBlock 为例）

ResNet 由多个残差块堆叠而成。BasicBlock 用于较浅的 ResNet（如 ResNet-18/34），结构如下：

```
        x
        │
   [Conv 3x3, BN, ReLU]
        │
   [Conv 3x3, BN]
        │
   ┌────┴────┐
   │         +
   │         │
   │      ReLU
   │         │
   └────> output
```

当输入输出通道数不同时，跳跃连接上需加一个 1x1 卷积调整维度。

### 3.4 与 VGG 的对比

- **VGG**：plain 网络，层数增加时训练困难。
- **ResNet**：引入残差连接，使得训练极深网络（如 ResNet-152）成为可能，且收敛更快。

### 3.5 PyTorch 代码示例（BasicBlock）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1  # 用于后续Bottleneck

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 主分支：两个3x3卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接：如果维度变化，用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)   # 残差连接
        out = F.relu(out)
        return out

# 测试
x = torch.randn(4, 64, 32, 32)
block = BasicBlock(64, 128, stride=2)
y = block(x)
print(y.shape)  # torch.Size([4, 128, 16, 16])
```

**关键点**：
- 跳跃连接将输入直接加到主分支输出上，要求两者形状一致（通道、高宽）。不一致时通过 1x1 卷积调整。
- 最后的 ReLU 在相加之后，保证输出非负（对于 ReLU 网络）。

---

## 总结

- **初始化**：根据激活函数选择 Xavier（tanh）或 Kaiming（ReLU），PyTorch 默认 Kaiming 但对 tanh 建议手动改。
- **归一化**：分清四种 Norm 的统计维度和适用场景，在代码中正确设置参数。
- **残差连接**：通过跳跃连接让网络学习残差，解决退化问题，实现简单且效果显著。

以上技巧是现代深度网络的基石，掌握它们能帮助你更快训练出有效模型。实践中多尝试、多调试，结合具体任务选择合适配置即可。