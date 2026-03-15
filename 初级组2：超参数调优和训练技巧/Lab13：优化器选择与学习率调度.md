## 深度学习训练技巧：优化器与学习率调度详解

在训练神经网络时，**优化算法**和**学习率调度**是两个至关重要的环节。它们直接影响模型的收敛速度、最终性能以及训练的稳定性。下面我将从底层原理（严谨 + 通俗类比）到 PyTorch 代码实现，为你逐一展开。

---

### 第一部分：优化器（Optimizer）

优化器的作用是根据损失函数的梯度更新网络的参数，让损失不断下降。最基础的是 SGD，然后衍生出 Momentum、Adam 等变体。

#### 1. SGD（随机梯度下降）

**底层原理**  
SGD 每次迭代用当前梯度直接更新参数：
\[
\theta_{t+1} = \theta_t - \eta \cdot g_t
\]
其中 \(\eta\) 是学习率，\(g_t\) 是当前 mini-batch 的梯度。

**通俗类比**  
想象你在山谷中，想走到最低点（最小化损失）。SGD 就是每次用脚感受一下当前位置的坡度（梯度），然后朝着最陡的下坡方向迈出一步，步长由学习率决定。缺点是容易在山谷两侧来回震荡，收敛慢。

#### 2. Momentum（动量）

**底层原理**  
Momentum 引入“速度”概念，将之前的梯度方向也考虑进来：
\[
v_t = \gamma v_{t-1} + \eta g_t
\]
\[
\theta_{t+1} = \theta_t - v_t
\]
其中 \(\gamma\) 通常取 0.9 左右。它累积了历史梯度的指数衰减平均，相当于给更新加了惯性。

**通俗类比**  
现在你不再是每一步都重新感受坡度，而是像一个从山顶滚下的球：球的速度会逐渐积累，即使经过平坦区域也能凭借惯性继续前进；遇到方向变化时，速度不会立刻反转，从而减小震荡。这能加速收敛并越过局部极小。

**PyTorch 代码**  
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

#### 3. Adam（自适应矩估计）

**底层原理**  
Adam 结合了 Momentum 和 RMSProp 的思想：它维护梯度的一阶矩（均值）和二阶矩（未中心化的方差），并自适应调整每个参数的学习率。
\[
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(一阶矩，类似动量)}
\]
\[
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(二阶矩，梯度平方的移动平均)}
\]
然后对 \(m_t\) 和 \(v_t\) 做偏差修正，最后更新参数：
\[
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\]
这样每个参数都有独立的自适应学习率：梯度变化大的参数学习率小，梯度稳定的参数学习率大。

**通俗类比**  
想象你在不同地形上行走：有的地方是沙地（梯度变化大），需要小步慢走；有的地方是硬地（梯度稳定），可以大步快走。Adam 会根据每个脚踩下去的软硬程度（梯度的方差）自动调整步长，同时保留动量。这使得 Adam 通常不需要精细调参，就能快速收敛。

**PyTorch 代码**  
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 常用 lr=1e-3
```

---

### 第二部分：学习率调度（Learning Rate Scheduling）

学习率是优化器中最重要的超参数。一开始我们可能用较大的学习率快速下降，但后期需要减小学习率以便精细收敛。**学习率调度**就是在训练过程中动态改变学习率。

#### 为什么需要调度？
- **学习率过大**：损失震荡，难以收敛到极小点。
- **学习率过小**：收敛极慢，可能陷入局部极小或鞍点。
- **动态调整**：前期大步探索，后期小步微调，兼顾速度与精度。

常见调度策略有阶梯下降、余弦退火、warmup 等。

#### 1. 阶梯式下降（Step Decay）

**原理**  
每隔一定 epoch，将学习率乘以一个衰减因子（如 0.1）。比如每 30 个 epoch 学习率减半。

**类比**  
就像爬楼梯：每到一个平台，你调整一下步伐，变得更小心，以免错过最低点。

**PyTorch 代码**  
```python
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 每30个epoch乘以0.1

for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()  # 每个epoch后更新学习率
```

#### 2. 余弦退火（Cosine Annealing）

**原理**  
学习率按照余弦函数从初始值下降到最小值（可以是 0 或其他设定值）。公式：
\[
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{T_{\text{cur}}}{T_{\max}}\pi))
\]
其中 \(T_{\max}\) 是总迭代次数（或周期长度），\(T_{\text{cur}}\) 是当前迭代。

**类比**  
想象一个弹簧振荡：学习率像余弦波一样平滑下降，然后在某个点再次升高（如果重启）或保持低位。平滑下降有助于避开尖锐的局部极小。

**PyTorch 代码**  
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)  # T_max 是半个周期长度
# 每个epoch后调用 scheduler.step()
```

#### 3. 带热启动的余弦退火（CosineAnnealingWarmRestarts）

**原理**  
这是余弦退火的改进版：学习率下降后，会突然“重启”到一个较高值，然后再次余弦下降。这可以跳出局部极小，多次探索。

**类比**  
就像跑步时每隔一段距离回到起点重新冲刺，但每次冲刺可能比上一次更深入。

**PyTorch 代码**  
```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0)
# T_0 是第一次重启的周期长度，T_mult 是每次周期延长的倍数
```

#### 4. Warmup（预热）

**原理**  
训练初期，模型参数随机，梯度不稳定，直接用大学习率可能造成振荡。Warmup 让学习率从 0 或很小的值线性（或非线性）增加到预设初始学习率，经过若干 epoch 后再使用正常调度。

**类比**  
就像汽车起步：先缓慢加速（warmup），等速度稳定后再全速行驶。

**PyTorch 代码**  
PyTorch 没有内置的 Warmup 调度器，但可以自定义或使用第三方库（如 `transformers`）。这里演示手动实现线性 warmup 与 StepLR 结合：

```python
from torch.optim.lr_scheduler import LambdaLR

def warmup_lambda(epoch):
    # epoch 从0开始，warmup_epochs=5
    if epoch < 5:
        return (epoch+1) / 5   # 线性从 0.2 到 1.0
    else:
        return 0.5 ** ((epoch-4)//30)  # 之后阶梯下降

scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
```

更常见的做法是**组合使用：先用 warmup，再用余弦退火。可以使用 `torch.optim.lr_scheduler.SequentialLR`（PyTorch 1.10+）将多个调度器串联。**

```python
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=50)

scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[5])
# milestones=[5] 表示在第5个epoch后切换到第二个调度器
```

---

### 综合示例：一个完整的训练脚本片段

下面展示如何将优化器与学习率调度组合使用，并在每个 epoch 后更新学习率。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, SequentialLR, LinearLR

# 假设已有模型、数据加载器等
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义调度器：先 warmup 5个epoch，然后余弦退火（周期10，重启因子2）
warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)
cosine = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # 验证
    model.eval()
    with torch.no_grad():
        # 计算验证损失/准确率
        pass
    
    # 更新学习率
    scheduler.step()
    
    # 打印当前学习率（可选）
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}, LR: {current_lr:.6f}")
```

---

### 总结与建议

- **优化器选择**：
  - SGD + Momentum 通常需要精细调参，但泛化能力可能更好。
  - Adam 自适应学习率，简单好用，收敛快，适合大多数任务。
  - 可以先用 Adam 快速得到基线，再用 SGD 调优。

- **学习率调度**：
  - 如果没有特殊需求，阶梯下降（StepLR）简单有效。
  - 余弦退火 + warmup 是现代训练（尤其是 Transformer 类模型）的标配，收敛效果更平滑。
  - Warmup 几乎总是有益的，特别是 batch size 较大时。

- **代码实现注意**：
  - 调度器的 `step()` 必须在每个 epoch 结束后调用（或每个 iteration，取决于调度器类型，如余弦重启需要按 iteration 更新）。
  - 可以随时用 `scheduler.get_last_lr()` 获取当前学习率。

希望这些详细解释和代码示例能帮你快速上手优化器与学习率调度的技巧。如果有不清楚的地方，欢迎继续提问！