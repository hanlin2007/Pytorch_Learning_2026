在单变量线性回归中，我们只用一个因素（学习时间）预测成绩。但现实生活中，一个输出结果往往由多个因素决定。多变量线性回归就是让模型学会**综合考虑多个因素**来做预测。

**多元线性回归：**
```
y = w₁x₁ + w₂x₂ + w₃x₃ + ... + wₙxₙ + b

矩阵表示：y = X @ W + b
```

#### 多变量线性回归实战：预测房价

定义模型结构：单层，多变量，线性全连接层
```python

class MultivariableLinearRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 线性层：输入input_dim维，输出1维
        # 这相当于 y = X @ W + b
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# 创建模型：输入特征数是3
model = MultivariableLinearRegression(num_features)

# 查看初始化参数
print(f"\n初始化参数：")
print(f"  w: {model.linear.weight.data.numpy()}")
print(f"  b: {model.linear.bias.data.item():.2f}")

```

定义损失函数和优化器
```python
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.0001)  # 学习率要调小，因为特征值范围大了
```

模型的训练
```python
epochs = 1000
losses = []

for epoch in range(epochs):
    # 前向传播
    predictions = model(X)
    loss = criterion(predictions, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        print(f"  当前w: {model.linear.weight.data.numpy().flatten()}")
        print(f"  当前b: {model.linear.bias.data.item():.2f}")

```

模型的测试
```python
print(f"真实规律：y = {true_w.numpy().flatten()} * X + {true_b.item()}")
print(f"学习结果：y = {model.linear.weight.data.numpy().flatten()} * X + {model.linear.bias.data.item():.2f}")

# 计算每个特征的误差
learned_w = model.linear.weight.data.numpy().flatten()
for i in range(num_features):
    feature_names = ['面积', '卧室数', '房龄']
    error = abs(learned_w[i] - true_w.numpy().flatten()[i])
    print(f"  特征{feature_names[i]}: 真实={true_w.numpy().flatten()[i]:.2f}, 学习={learned_w[i]:.2f}, 误差={error:.2f}")
```


在多变量回归中，不同特征的数值范围可能差异很大，会导致问题：
- **权重更新不平衡**：大数值特征（面积）的梯度会很大，小数值特征（卧室数）的梯度会很小
- **收敛缓慢**：优化器需要小心翼翼地调整学习率

**解决方案：特征缩放**，两种常用方法：

```python
# 1. 标准化（Standardization）：将数据变成均值为0，标准差为1
def standardize(x):
    return (x - x.mean(dim=0)) / x.std(dim=0)

# 2. 归一化（Normalization）：将数据缩放到[0,1]区间
def normalize(x):
    return (x - x.min(dim=0)[0]) / (x.max(dim=0)[0] - x.min(dim=0)[0])

# 实战中使用：
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_numpy = X.numpy()
X_scaled = scaler.fit_transform(X_numpy)  # 计算均值和标准差，并转换
X_scaled = torch.from_numpy(X_scaled).float()

# 注意：用训练集的均值和标准差来转换测试集！
X_test_scaled = scaler.transform(X_test.numpy())
```
注意区分正则化！！
