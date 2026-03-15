### 1. L2 正则化（权重衰减）

#### 原理
- **严谨原理**：在损失函数中增加所有权重的平方和乘以一个系数 λ/2，即  
  \[
  L_{\text{总}} = L_{\text{原始}} + \frac{\lambda}{2} \sum_{i} w_i^2
  \]  
  梯度下降时，权重更新规则变为 \( w \leftarrow w - \eta \cdot (\frac{\partial L_{\text{原始}}}{\partial w} + \lambda w) \)，相当于每次更新时先把权重缩小一点点（乘以 \(1 - \eta \lambda\)），再沿梯度方向更新。这就迫使权重保持较小的值。
- **通俗类比**：想象你在用橡皮筋拉着一组小滑块（权重）。橡皮筋的弹性会阻止滑块跑得太远，滑块越小，橡皮筋越松。L2 正则化就像是给每个权重绑了一根橡皮筋，不让它们过大。

#### PyTorch 实现
在 PyTorch 中，L2 正则化通过优化器的 `weight_decay` 参数实现，`weight_decay` 就是公式中的 λ。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的模型
model = nn.Linear(10, 2)  # 输入10维，输出2维

# 使用 SGD 优化器，并设置 weight_decay=0.01（λ=0.01）
optimizer = optim.SGD(model.parameters(), lr=0.1, weight_decay=0.01)

# 训练循环中正常使用即可
for epoch in range(100):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()  # 这一步会自动加上 L2 惩罚
```

**注意**：`weight_decay` 在 PyTorch 的 SGD 和 Adam 等优化器中都可用，它默认对所有权重（不包括偏置）施加 L2 惩罚，因为偏置通常不需要正则化。如果你想自定义，可以分组传递参数。

---

### 2. L1 正则化

#### 原理
- **严谨原理**：在损失函数中加入所有权重的绝对值之和乘以 λ，即  
  \[
  L_{\text{总}} = L_{\text{原始}} + \lambda \sum_{i} |w_i|
  \]  
  由于绝对值函数在 0 处不可导，优化时会使得部分权重正好变成 0，从而让模型变稀疏（很多特征被忽略）。
- **通俗类比**：L1 正则化好比给你的模型发“零花钱”，每保留一个权重就要花一份钱。为了省钱，模型会尽量把没用的权重直接砍掉（设为 0），只留下最重要的几个。

#### PyTorch 实现
PyTorch 没有直接提供 L1 惩罚的优化器参数，需要手动把 L1 损失加到总损失中。

```python
import torch
import torch.nn as nn

model = nn.Linear(10, 2)
criterion = nn.MSELoss()  # 假设是回归任务

# 超参数 λ
l1_lambda = 0.001

for epoch in range(100):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, targets)
    
    # 计算 L1 惩罚项
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    loss = loss + l1_lambda * l1_norm
    
    loss.backward()
    optimizer.step()
```

**说明**：`model.parameters()` 返回所有权重和偏置，如果只想对权重加 L1，可以筛选 `p.dim() > 1` 的参数（通常偏置是一维）。

---

### 3. Dropout（随机失活）

#### 原理
- **严谨原理**：在训练过程中，以概率 \(p\) 随机将某些神经元的输出置为 0（失活），相当于每次前向传播都在训练一个不同的子网络。测试时所有神经元都参与，但输出要乘以 \(1-p\)（或使用 “inverted dropout” 自动缩放）以保证期望一致。这迫使网络不依赖特定神经元，从而增强鲁棒性。
- **通俗类比**：公司开会时，随机让一半员工休息，剩下的人必须独立完成任务。这样大家就不会过度依赖某个“明星员工”，每个人都能独当一面，整个团队就更稳定。

#### PyTorch 实现
使用 `torch.nn.Dropout` 层，参数 `p` 是**失活概率**（即每个神经元被丢弃的概率）。

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(p=0.5)  # 50% 神经元失活
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)   # 训练时随机丢弃，测试时自动变成恒等映射
        x = self.fc2(x)
        return x

model = MyModel()
# 训练时 model.train()，测试时 model.eval()，Dropout 会自动切换行为
```

**重要**：在训练循环开始前记得调用 `model.train()`，测试/验证前调用 `model.eval()`。

---

### 4. 早停（Early Stopping）

#### 原理
- **严谨原理**：在训练过程中监控验证集上的性能（如损失或准确率），当验证性能连续若干轮（patience）不再提升时，就停止训练，并回滚到验证性能最好的模型参数。这样可以避免训练过度导致过拟合。
- **通俗类比**：你在背单词，背完一遍后做模拟题，如果连续几次模拟题成绩都没有提高，说明再背下去可能只是死记硬背（过拟合），不如就此打住。

#### PyTorch 实现
需要手动编写逻辑，通常使用一个计数器记录验证损失未下降的轮数。

```python
best_val_loss = float('inf')
patience = 5          # 允许连续多少次不降低
counter = 0
best_model_state = None

for epoch in range(epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()  # 保存最好参数
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping!")
            model.load_state_dict(best_model_state)  # 恢复最好参数
            break
```

**扩展**：可以使用 `torch.save` 把 `best_model_state` 存到文件，防止意外中断。

---

### 总结：如何组合使用这些技巧？

在实际项目中，通常组合多种正则化方法：

- **L2 正则化**：几乎必加，通过优化器的 `weight_decay` 轻松实现，建议值 `1e-4` ~ `1e-2`。
- **Dropout**：放在全连接层之间，常用概率 `0.2~0.5`，卷积层后有时也用 `0.1` 左右的 Spatial Dropout。
- **早停**：始终使用，防止浪费时间，`patience` 一般取 5~20。
- **数据增强**：适用于图像、音频等数据，可大幅提升泛化能力。
- **L1 正则化**：当你希望模型稀疏（如特征选择）时使用，一般 λ 较小。

在训练时，按以下模式组织代码：

```python
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # 如果有 L1 惩罚，在这里加上
        loss.backward()
        optimizer.step()
    
    model.eval()
    val_loss = validate(model, val_loader)
    scheduler.step(val_loss)
    
    # 早停逻辑
    if early_stopping(val_loss):
        break
```
