import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 定义 LeNet 网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 第一层卷积层: 输入通道数 1，输出通道数 6，卷积核大小 5x5，步长 1，填充 2
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # 第一层池化层: 平均池化，池化核大小 2x2，步长 2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # 第二层卷积层: 输入通道数 6，输出通道数 16，卷积核大小 5x5，步长 1
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # 第二层池化层: 平均池化，池化核大小 2x2，步长 2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # 全连接层 1: 输入特征数 16*6*6，输出特征数 120
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        # 批标准化层 1
        self.bn1 = nn.BatchNorm1d(120)
        # Dropout 层 1，防止过拟合
        self.dropout1 = nn.Dropout(0.5)
        # 全连接层 2: 输入特征数 120，输出特征数 84
        self.fc2 = nn.Linear(120, 84)
        # 批标准化层 2
        self.bn2 = nn.BatchNorm1d(84)
        # Dropout 层 2，防止过拟合
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层 3: 输入特征数 84，输出特征数 10
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # 通过第一个卷积层和池化层
        x = self.pool1(torch.relu(self.conv1(x)))
        # 通过第二个卷积层和池化层
        x = self.pool2(torch.relu(self.conv2(x)))
        # 展平操作
        x = x.view(-1, 16 * 6 * 6)
        # 通过第一个全连接层和批标准化层，再应用ReLU激活函数
        x = torch.relu(self.bn1(self.fc1(x)))
        # Dropout 层
        x = self.dropout1(x)
        # 通过第二个全连接层和批标准化层，再应用ReLU
        x = torch.relu(self.bn2(self.fc2(x)))
        # Dropout 层
        x = self.dropout2(x)
        # 通过第三个全连接层（输出层）
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    # 检查是否有可用的 GPU，有则使用 CUDA 加速
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    # 实例化网络并将其移动到设备（CPU 或 GPU）
    net = LeNet().to(device)

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为单通道灰度图像
        transforms.Resize((32, 32)),  # 调整图像大小为 32x32
        transforms.RandomRotation(15),  # 随机旋转，范围为 -15 到 15 度
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化到 [-1, 1]
    ])

    train_data_path = 'E:/Python-projects/pytorch/练习/项目实战/mnist_images/train'
    test_data_path = 'E:/Python-projects/pytorch/练习/项目实战/mnist_images/test'

    # 加载训练和测试数据集
    trainset = ImageFolder(root=train_data_path, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = ImageFolder(root=test_data_path, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 调整学习率为 0.001

    # 记录损失
    train_losses = []

    # 训练网络
    for epoch in range(25):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 获取输入数据并将其移动到设备（CPU 或 GPU）
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = net(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()

            # 累加损失
            running_loss += loss.item()
        
        # 记录当前 epoch 的损失
        train_losses.append(running_loss / len(trainloader))
        print(f'[Epoch {epoch + 1}] loss: {running_loss / len(trainloader):.3f}')
        
    print("Finished Training")

    # 绘制损失函数的训练下降图
    plt.plot(train_losses, label='Training loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 测试网络性能
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct / total}%')

    # 保存模型
    PATH = 'E:/Python-projects/pytorch/lenet.pth'
    torch.save(net.state_dict(), PATH)
