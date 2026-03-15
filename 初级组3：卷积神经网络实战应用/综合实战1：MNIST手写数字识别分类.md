
### （一）导入模块 
- time模块记录下载数据集和训练的总时间 
- s.path管理数据路径
- 导入tqdm库用于展示进度条
```python
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

import numpy as np
import time
import os
```

### （二）配置与数据加载有关的超参数
 
```python
class Config:
    # 数据参数
    data_path = './data'           # 数据存储路径
    batch_size = 64                # 批次大小
    num_workers = 0                # 数据加载线程数（Windows建议0）
    
    # 模型参数
    input_size = 784               # 输入特征数 (28 * 28)
    hidden_size = 128              # 隐藏层神经元数
    output_size = 10               # 输出类别数 (0-9)
    dropout_rate = 0.2             # Dropout比率
    
    # 训练参数
    epochs = 10                    # 训练轮数
    learning_rate = 0.001          # 学习率
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 保存路径
    model_save_path = './mnist_model.pth'

config = Config()

# 打印配置信息
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
print(f"训练设备: {config.device}")
```

### (三) MINST数据集准备

[[Lab04：真实数据集加载]]

```python

def prepare_data():
    # 定义数据预处理转换
    # ToTensor(): 将PIL Image转换为Tensor
    #           并将像素值从[0,255]归一化到[0.0,1.0]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))  # 可选：标准化
    ])
    
    print("\n正在加载MNIST数据集...")
    start_time = time.time()
    
    # 加载训练集
    train_dataset = torchvision.datasets.MNIST(
        root=config.data_path,
        train=True,
        download=True,           #参考前面Data Loading部分，自动下载数据集
        transform=transform
    )
    
    # 加载测试集
    test_dataset = torchvision.datasets.MNIST(
        root=config.data_path,
        train=False,
        download=True,
        transform=transform
    )
    
    load_time = time.time() - start_time
    print(f"数据集加载完成! 耗时: {load_time:.2f}秒")
    print(f"训练样本数: {len(train_dataset)}")
    print(f"测试样本数: {len(test_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,            # 打乱训练数据
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,           # 测试集不需要打乱
        num_workers=config.num_workers,
        pin_memory=True if config.device == 'cuda' else False
    )
    
    return train_loader, test_loader, train_dataset, test_dataset
```

### (四) 定义神经网络结构

```python

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate):
        super(MLP, self).__init__()
        
        self.model = nn.Sequential(
            # 第一层：全连接 + ReLU激活
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 防止过拟合
            
            # 第二层：全连接 + ReLU激活
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 输出层：全连接（无激活，后面用CrossEntropyLoss）
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        # 将28x28的图像展平为784维向量
        x = x.view(-1, config.input_size)
        return self.model(x)

```

### （五）训练框架

```python
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()  # 设置为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 将数据移动到GPU
        images = images.to(config.device)
        labels = labels.to(config.device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()        # 计算梯度
        optimizer.step()       # 更新参数
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 每100个batch打印一次进度
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{config.epochs}], '
                  f'Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy

```

### （六）测试模块：测试时改为.eval()，同时不计算梯度

```python
def test(model, test_loader, criterion):
    model.eval() 
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy
```


### （七）主函数入口

```python
def main():
    print("开始MNIST手写数字识别训练")
    
    # 1. 准备数据
    train_loader, test_loader, train_dataset, test_dataset = prepare_data()
    
    # 2. 初始化模型
    print("\n模型初始化")
    model = MLP(
        input_size=config.input_size,
        hidden_size=config.hidden_size,
        output_size=config.output_size,
        dropout_rate=config.dropout_rate
    ).to(config.device)
    
    # 打印模型结构
    print(f"\n模型结构:\n{model}")
    
    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # 4. 记录训练过程
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_test_accuracy = 0.0
    
    # 5. 训练循环
    print("\n开始训练：")
    total_start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        
        # 测试
        test_loss, test_acc = test(model, test_loader, criterion)
        
        # 记录
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        epoch_time = time.time() - epoch_start_time
        
        print(f'\nEpoch [{epoch+1}/{config.epochs}] 完成, 耗时: {epoch_time:.2f}秒')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.2f}%')
        
        # 保存最佳模型
        if test_acc > best_test_accuracy:
            best_test_accuracy = test_acc
            torch.save(model.state_dict(), config.model_save_path)
            print(f'模型已保存，测试准确率: {best_test_accuracy:.2f}%')
    
    total_time = time.time() - total_start_time
    print(f"\n训练完成，总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"最佳测试准确率: {best_test_accuracy:.2f}%")
    
    # 函数运行入口
if __name__ == '__main__':
    main()
```