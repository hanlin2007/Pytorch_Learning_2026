import torch  # 注意这个包的名称，不是pytorch，后面都是torch.nn等
import torch.nn as nn
import torch.optim as optim

# 第一部分：准备训练数据
# 这里使用手动设置，将张量设置在GPU上
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


class LinearRegressionModel(nn.Module):  
    def __init__(self):  
        super().__init__()    
        # 定义一个线性层 
        self.linear = nn.Linear(1,1)    

    def forward(self,x):
        return self.linear(x)          
        
# 创建模型实例
model = LinearRegressionModel()
# 记得添加这一行，把模型移到GPU
model = model.to('cuda')
print(f"initial parameters show:w={model.linear.weight.item():.2f},b={model.linear.bias.item():.2f}")

# 第三部分：定义损失函数和优化器
# 损失函数：均方误差（Mean Squared Error, MSE）
criterion = nn.MSELoss()
# 优化器：随机梯度下降（Stochastic Gradient Descent, SGD）
# 参数：model.parameters()就是模型要学习的w和b
# lr=0.01是学习率，控制每次调整的步伐大小（太大容易跑偏，太慢容易慢）
optimizer = optim.SGD(model.parameters(), lr=0.01)

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

# 第五部分：测试模型
with torch.no_grad():  # 测试时不需要计算梯度，节省内存
    # 记得把测试用的张量也移动到GPU上，否则训练张量、权重参数、测试张量位置不统一会报错
    x_test = torch.tensor([[5.0], [6.0]],device='cuda')
    y_pred = model(x_test)
    print(f"\n预测结果：")
    print(f"x=5 → y={y_pred[0].item():.2f} (应该接近11)")
    print(f"x=6 → y={y_pred[1].item():.2f} (应该接近13)")
