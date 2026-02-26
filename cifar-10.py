"""
CIFAR-10 物体识别
类别：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# ======================
# 1. 数据预处理与加载
# ======================

# 数据增强（CIFAR-10需要更强的数据增强）
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(),      # 随机翻转
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# ======================
# 2. 只有在 __main__ 里才执行以下代码（Windows 必须！）
# ======================

def main():
    # 加载CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    # 划分验证集: 45000 训练，5000 验证
    train_size = 45000
    val_size = 5000
    trainset, valset = torch.utils.data.random_split(trainset, [train_size, val_size])

    # DataLoader 设置 num_workers（Windows 必须放在 main() 里！）
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=128, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # ======================
    # 3. 模型定义
    # ======================
    class CIFAR10CNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: 32x32
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 16x16

                # Block 2: 16x16
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 8x8

                # Block 3: 8x8
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 4x4
            )

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Linear(256, num_classes)
            )

        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x

    # ======================
    # 4. 训练准备
    # ======================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    model = CIFAR10CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # ======================
    # 5. 训练函数
    # ======================
    def train_cifar10(model, trainloader, valloader, epochs=50):
        best_acc = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            correct = 0
            total = 0

            pbar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{epochs}')
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })

            # 验证
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, targets in valloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

            # 记录
            train_acc = 100. * correct / total
            val_acc = 100. * val_correct / val_total

            train_losses.append(train_loss / len(trainloader))
            val_losses.append(val_loss / len(valloader))
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            # 学习率调整
            scheduler.step()

            print(f'Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, LR: {scheduler.get_last_lr()[0]:.6f}')

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'cifar10_best.pth')
                print(f'✓ 保存最佳模型，准确率: {val_acc:.2f}%')

        return train_losses, val_losses, train_accs, val_accs

    # ======================
    # 6. 开始训练（只在 main() 里运行！）
    # ======================
    history = train_cifar10(model, trainloader, valloader, epochs=50)

    # ======================
    # 7. 测试（加载最佳模型）
    # ======================
    model.load_state_dict(torch.load('cifar10_best.pth'))
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()

    print(f'\n测试集准确率: {100. * test_correct / test_total:.2f}%')

# ======================
# 8. Windows 必须：用 if __name__ == '__main__' 包裹主程序
# ======================
if __name__ == '__main__':
    main()