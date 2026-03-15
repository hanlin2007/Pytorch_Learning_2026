
把逻辑回归拓宽到多元分类，就是Softmax
#### **第一部分：深度学习基本原理速成**

**1. Softmax函数：公平的“翻译官”**

Sigmoid 是“二选一”的翻译官，而 Softmax 是“多选一”的翻译官。

- **公式**: S(zi​)=∑j=1K​ezj​ezi​​
    
    - K是类别的总数（比如3）。
        
    - zi​是模型为第 i个类别计算出的原始分数（logit）。
    
- **作用**:
    
    1. **指数化 (Exponentiation)**: 对所有的 logits 进行 ez运算。这有两个效果：一是确保所有值为正，二是**放大利好，缩小利空**。比如，如果“狗”的 logit 是 2，“猫”的 logit 是 1，指数化后，“狗”的优势就从 2倍 变成了 e2≈7.4倍，这使得模型对“最有把握”的类别更加自信。
        
    2. **归一化 (Normalization)**: 将所有指数化后的值求和，然后用每个值除以总和。这确保了所有类别的概率之和为 1。    

**直观理解**：Softmax 把模型算出的“原始分数”转换成了“概率分布”，清晰地展示了模型对每个选项的信心。

**2. 怎么判断“猜得对不对”？—— 多分类交叉熵**

二分类的交叉熵只关心“是”和“非”两个选项。多分类的交叉熵则关心“正确答案”和“所有其他错误答案”的差距。

- **公式**: L=−N1​∑i=1N​∑c=1K​yic​⋅log(y^​ic​)
    
    - K是类别总数。
        
    - yic​是**one-hot编码**的标签。如果第 i个样本的真实类别是 c，那么 yic​=1，否则为 0。
        
        - 例如，样本是“狗”，类别顺序是 \[猫, 狗, 鸟]，那么它的 one-hot 标签就是 `[0, 1, 0]`
    - y^​ic​是模型预测该样本属于类别 c的概率。


**通俗理解**：

- 对于那个“狗”的样本，其 one-hot 标签是 `[0, 1, 0]`。
    
- 公式里，只有 c=2（“狗”这个类别）的那一项 yic​是 1，其他都是 0。
    
- 所以，这个样本的损失就简化为：L=−log(y^​狗​)。
    
- 这和我们二分类时，当 y=1的情况一模一样！
    
- **核心思想**：**多分类交叉熵，本质上就是“只关注正确答案的概率，并希望它越大越好”**。如果模型对正确答案的预测概率 y^​正确答案​接近 1，那么 −log(1)=0，损失极小。如果它接近 0，那么 −log(0)会是一个巨大的数，损失巨大。
    

**4. `CrossEntropyLoss`**

和 `BCEWithLogitsLoss`一样，PyTorch 也为多分类问题提供了一个“二合一”的、稳定且高效的损失函数：`nn.CrossEntropyLoss`。

**`nn.CrossEntropyLoss`的黄金法则：**

> **它期望的输入是“生肉”（logits），并且它内部已经集成了 `LogSoftmax`和 `NLLLoss`（负对数似然损失）两个步骤。因此，你绝对不能在模型定义里再加 `Softmax`层！**

- **模型定义 (`forward`函数)**: 只做线性变换，输出 logits。
    
    - `out = self.linear(x)`# 输出形状: `[batch_size, num_classes]`
        
    
- **损失计算**: 直接将 logits 和 one-hot 标签（或类别索引）喂给 `CrossEntropyLoss`。
    
    - `loss = criterion(logits, labels)`
        
    
- **预测时**: 如果你想要看概率，必须**手动**在 logits 上应用 `F.softmax()`。
    
    - `probabilities = F.softmax(logits, dim=1)`


**为什么这样做？**

- **数值稳定性**: 和 `BCEWithLogitsLoss`的原因一样，内部合并计算更稳定。
    
- **一致性**: 保持了训练和预测流程的清晰分离。


---

#### **第二部分：PyTorch 实战代码 (含详细注释)**

让我们用一个三分类的例子（区分猫、狗、鸟）来巩固这个概念。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F  # 导入 functional，里面包含 softmax 等函数

# --- 0. 超参数与设备设置 ---
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- 1. 准备数据 (三分类) ---
# 假设我们有 300 个样本，每个样本有 2 个特征
num_samples = 300
num_classes = 3  # 0: Cat, 1: Dog, 2: Bird

X = torch.randn(num_samples, 2, dtype=torch.float32, device=device)

# 生成标签：根据特征区域划分，制造一个可分的数据集
# 区域1 (x1+x2<=-1): Cat
# 区域2 (-1<x1+x2<=1): Dog
# 区域3 (x1+x2>1): Bird
y_labels = torch.zeros(num_samples, dtype=torch.long, device=device)  # 初始全为0 (Cat)
dog_mask = (X[:, 0] + X[:, 1] > -1) & (X[:, 0] + X[:, 1] <= 1)
bird_mask = X[:, 0] + X[:, 1] > 1
y_labels[dog_mask] = 1  # Dog
y_labels[bird_mask] = 2  # Bird

# 创建数据加载器
dataset = TensorDataset(X, y_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- 2. 定义模型 (严格遵循“最后一层不加Softmax”) ---
class SoftmaxClassifierModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # 一个线性层，输出维度是类别数量
        self.linear = nn.Linear(input_dim, num_classes)
        # 注意：没有 self.softmax = nn.Softmax(dim=1) 了！

    def forward(self, x):
        # 前向传播：只做线性变换，输出 logits
        logits = self.linear(x)
        return logits  # 形状: [batch_size, num_classes]

# 实例化模型
model = SoftmaxClassifierModel(input_dim=2, num_classes=num_classes).to(device)
print("--- Model Architecture ---")
print(model)


# --- 3. 定义损失函数和优化器 ---
# 【关键点】使用 CrossEntropyLoss，因为它期望的输入是 logits
# 它会自动处理 logits -> log_softmax -> nll_loss
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.1)  # Adam通常比SGD收敛更快


# --- 4. 训练循环 ---
num_epochs = 100

print("\n--- Training Start ---")
for epoch in range(num_epochs):
    epoch_loss = 0.0
    correct = 0
    total = 0

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        # 1. 前向传播
        # model(data) 输出的是 logits
        logits = model(data)

        # 计算损失
        # CrossEntropyLoss 接收 logits 和 类别索引 (LongTensor)
        loss = criterion(logits, labels)

        # 2. 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        epoch_loss += loss.item()

        # 计算准确率 (需要用 softmax 转换后才能比较概率，或直接用 argmax)
        # 方法一：用 softmax + argmax (更直观)
        probabilities = F.softmax(logits, dim=1)
        _, predicted_labels = torch.max(probabilities, 1)
        
        # 方法二（更高效）：直接用 argmax on logits (因为softmax不改变最大值的位置)
        # _, predicted_labels = torch.max(logits, 1)

        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()

    avg_loss = epoch_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

print("--- Training Finished ---")


# --- 5. 预测与可视化 ---
print("\n--- Prediction Demo ---")
model.eval()

with torch.no_grad():
    # 生成网格数据进行可视化
    xx, yy = torch.meshgrid(torch.linspace(-4, 4, 200), torch.linspace(-4, 4, 200), indexing='xy')
    grid_data = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1).float().to(device)
    
    # 模型输出 logits
    logits_on_grid = model(grid_data)
    # 手动应用 softmax 得到概率分布
    probs_on_grid = F.softmax(logits_on_grid, dim=1)
    # 获取预测的类别
    preds_on_grid = torch.argmax(probs_on_grid, dim=1).view(xx.shape)

# 打印几个具体点的预测
test_points = torch.tensor([[2.0, 2.0], [-2.0, 0.0], [0.0, 0.0]], device=device)
with torch.no_grad():
    test_logits = model(test_points)
    test_probs = F.softmax(test_logits, dim=1)  # 预测时，我们手动用softmax
    test_preds_indices = torch.argmax(test_probs, dim=1)
    class_names = ['Cat', 'Dog', 'Bird']
    test_preds_names = [class_names[p.item()] for p in test_preds_indices]

print("Test Point Predictions:")
for i in range(len(test_points)):
    point = test_points[i].cpu().tolist()
    probs = test_probs[i].cpu().tolist()
    pred_name = test_preds_names[i]
    print(f"  Point: {point}")
    print(f"    Probabilities: Cat={probs[0]:.4f}, Dog={probs[1]:.4f}, Bird={probs[2]:.4f}")
    print(f"    Predicted: {pred_name}")

# 可视化
try:
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制数据点
    ax.scatter(X[y_labels==0, 0].cpu(), X[y_labels==0, 1].cpu(), c='blue', label='Cat', edgecolors='k', alpha=0.7)
    ax.scatter(X[y_labels==1, 0].cpu(), X[y_labels==1, 1].cpu(), c='red', label='Dog', edgecolors='k', alpha=0.7)
    ax.scatter(X[y_labels==2, 0].cpu(), X[y_labels==2, 1].cpu(), c='green', label='Bird', edgecolors='k', alpha=0.7)
    
    # 绘制决策区域
    cmap = plt.cm.get_cmap('viridis', num_classes)
    ax.contourf(xx.cpu(), yy.cpu(), preds_on_grid.cpu(), alpha=0.3, cmap=cmap)
    # 添加颜色条以显示类别
    # cbar = plt.colorbar(ax.collections[0], ticks=[0.17, 0.5, 0.83]) # 大致位置
    # cbar.ax.set_yticklabels(['Cat', 'Dog', 'Bird'])
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('Softmax Classifier Decision Boundaries')
    ax.legend()
    plt.show()

except ImportError:
    print("\nMatplotlib not found. Skipping visualization.")
```

**代码关键点复盘：**

1. **模型输出**: `forward`返回 `self.linear(x)`，输出 logits，形状为 `[batch_size, num_classes]`。
    
2. **损失函数**: 使用 `nn.CrossEntropyLoss(logits, labels)`。这里的 `labels`是类别索引（`LongTensor`），而不是 one-hot 向量。PyTorch 会自动处理。
    
3. **预测时获取概率**: 必须使用 `F.softmax(logits, dim=1)`。`dim=1`表示在类别维度上进行归一化。
    
4. **预测时获取类别**: 使用 `torch.argmax(probabilities, dim=1)`找到概率最大的类别索引。由于 Softmax 不改变最大值的位置，也可以直接用 `torch.argmax(logits, dim=1)`，效率更高。