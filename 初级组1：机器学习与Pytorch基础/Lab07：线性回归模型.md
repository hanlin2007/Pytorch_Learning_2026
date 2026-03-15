
第一次尝试实战，来手写第一个Pytorch框架，训练一个线性回归模型

```python
import torch  # 注意这个包的名称，一开始写成了pytorch，torch.nn等全都是torch
import torch.nn as nn
import torch.optim as optim
  

# 第一部分：准备训练数据
x_train = torch.tensor([[1.0],[2.0],[3.0],[4.0]],device=torch.device('cuda')) # 输入学习时间
y_train = torch.tensor([[3.0],[5.0],[7.0],[9.0]],device=torch.device('cuda')) # 输出实际分数

# 查看GPU数量
print(f"GPU count: {torch.cuda.device_count()}")

# 查看当前使用的GPU名称
print(f"current GPU: {torch.cuda.get_device_name(0)}")

# 查看所有GPU名称
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

print(x_train)
print(y_train)
print(x_train.device)
print(y_train.device)
print(f"shape show:x={x_train.shape},y={y_train.shape}")
```

```
GPU count: 1
current GPU: NVIDIA GeForce RTX 4050 Laptop GPU
GPU 0: NVIDIA GeForce RTX 4050 Laptop GPU
tensor([[1.],
        [2.],
        [3.],
        [4.]], device='cuda:0')
tensor([[3.],
        [5.],
        [7.],
        [9.]], device='cuda:0')
# 这个cuda:0不是没有cuda，而是默认使用的GPU的编号，在torch.device('cuda') 中可以手动指定
cuda:0   
cuda:0  
shape show:x=torch.Size([4, 1]),y=torch.Size([4, 1])
```

默认创建的张量都是简化在cpu中的，需要手动指明使用GPU来训练，可以后续移动，也可以创建时直接使用cuda

```python

# 第二部分：定义模型结构
class LinearRegressionModel(nn.Module):  
    def __init__(self):  
        super().__init__()    
        # 定义一个线性层 
        self.linear = nn.Linear(1,1)    
        
    def forward(self,x);        # 注意forward函数不需要添加双下划线
        return self.linear(x)          
        
# 创建模型实例
model = LinearRegressionModel()
print(f"initial parameters show:w={model.linear.weight.item():.2f},b={model.linear.bias.item():.2f}")
```

==在实现的过程中，发现这里产生了很多问题==：

1.突然发现了PyTorch 编程的核心思想！！

底层的函数全部都使用pytorch库封装好了，比如各种神经网络层（全连接层、卷积层、池化层、循环层、子注意力层等等）、数据加载工具、训练、自动求导、优化、保存与加载工具等，只需要调用和调整网络的结构就可以了

这是一个通用的训练框架：
```python
for epoch in range(num_epochs):
    for batch_data, batch_labels in dataloader:
        outputs = model(batch_data)          # 前向传播
        loss = criterion(outputs, batch_labels)  # 计算损失
        optimizer.zero_grad()                # 清空梯度
        loss.backward()                      # 反向传播
        optimizer.step()                     # 更新参数
```

因此，利用PyTorch，我们需要修改和优化的重点在于：
理解各种层和结构的使用情景（什么时候用 `Linear`？什么时候用 `Conv2d+MaxPool`？）
掌握网络结构设计（层与层之间如何连接，层数过大可能导致梯度消失/爆炸）
学会**调参和优化**（后续！）

2.python语法之基于类创建对象
`model = LinearRegressionModel()`
`self.linear = nn.Linear(1,1)`
这两个部分都是基于类创建对象，而不是调用成员方法，至于对象，后续还可以传入其他参数，在PyTorch中多为`input_tensor`参数，比如`model(x)`以及`linear(x)` 均为这一用法


3.为什么后续的前向传播没有调用`forward()`函数，而是直接使用了`model(x)`?
`model(x)`的实际执行代码是魔法方法 `model.__call__(x)`
`nn.Module`中有对于`__call__`的定义，其中调用了`forward`函数，因此只用隐式调用就可以进行一次前向传播的计算

```python
# 第三部分：定义损失函数和优化器
# 损失函数：均方误差（Mean Squared Error, MSE）
criterion = nn.MSELoss()
# 优化器：随机梯度下降（Stochastic Gradient Descent, SGD）
# 参数：model.parameters()就是模型要学习的w和b
# lr=0.01是学习率，控制每次调整的步伐大小（太大容易跑偏，太慢容易慢）
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

```python
# 第四部分：训练模型 
epochs = 1000  
for epoch in range(epochs):
    # 前向传播：计算预测值
    predictions = model(x_train)
    
    # 计算损失：看预测得有多准
    loss = criterion(predictions, y_train)
    
    # 反向传播：最关键的三行代码！
    optimizer.zero_grad()  # 1. 清空之前的梯度（防止累积）
    loss.backward()        # 2. 反向传播，计算梯度（自动微分！）
    optimizer.step()       # 3. 更新参数：w = w - lr * gradient
    
    # 每100轮打印一次进度
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
        print(f'    w={model.linear.weight.item():.2f}, b={model.linear.bias.item():.2f}')
```

```python
# 第五部分：测试模型
with torch.no_grad():  # 测试时不需要计算梯度，节省内存
    x_test = torch.tensor([[5.0], [6.0]])
    y_pred = model(x_test)
    print(f"\n预测结果：")
    print(f"x=5 → y={y_pred[0].item():.2f} (应该接近11)")
    print(f"x=6 → y={y_pred[1].item():.2f} (应该接近13)")
```

```
initial parameters show:w=-0.04,b=0.87
Epoch [100/1000], Loss: 0.0231
    w=1.87, b=1.37
Epoch [200/1000], Loss: 0.0127
    w=1.91, b=1.27
Epoch [300/1000], Loss: 0.0070
    w=1.93, b=1.20
Epoch [400/1000], Loss: 0.0038
    w=1.95, b=1.15
Epoch [500/1000], Loss: 0.0021
    w=1.96, b=1.11
Epoch [600/1000], Loss: 0.0012
    w=1.97, b=1.08
Epoch [700/1000], Loss: 0.0006
    w=1.98, b=1.06
Epoch [800/1000], Loss: 0.0003
    w=1.98, b=1.05
Epoch [900/1000], Loss: 0.0002
    w=1.99, b=1.03
Epoch [1000/1000], Loss: 0.0001
    w=1.99, b=1.02

预测结果：
x=5 → y=10.98 (应该接近11)
x=6 → y=12.97 (应该接近13)
```

==我老是急着想用GPU来训练模型，但是太急了反而过程中遇到了很多问题==

如果希望使用GPU来训练和保存模型参数，必须保证，训练张量、模型权重参数、测试张量全部基于GPU，需要做以下统一
```python
# 创建模型实例
model = LinearRegressionModel()

# ✅ 添加这一行，把模型移到GPU
model = model.to('cuda')

print(f"initial parameters show:w={model.linear.weight.item():.2f},b={model.linear.bias.item():.2f}")
```
```python
with torch.no_grad():
    x_test = torch.tensor([[5.0], [6.0]], device='cuda')  # 加上 device='cuda'
    y_pred = model(x_test)
   
```

也可以采用默认再移动的方式
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1）数据：先按普通方式创建，再统一移动
x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # 默认在 CPU
y_train = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

x_train = x_train.to(device)
y_train = y_train.to(device)

# 2）模型：先创建，再 .to(device)
model = LinearRegressionModel()      # 默认在 CPU
model = model.to(device)            # 移到 GPU

# 3）测试数据同理
x_test = torch.tensor([[5.0], [6.0]])
x_test = x_test.to(device)          # 也要移过去
```
