import torch
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import torchvision.transforms as transforms  # 导入图像变换模块
from torchvision.datasets import ImageFolder  # 导入数据集模块
from torch.utils.data import DataLoader  # 导入数据加载器模块
import matplotlib.pyplot as plt  # 导入绘图模块

# 定义LeNet神经网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 定义第一个卷积层，输入通道为1，输出通道为6，卷积核大小为5x5，步长为1，填充为2
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        # 定义第一个池化层，使用平均池化，池化核大小为2x2，步长为2，无填充
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # 定义第二个卷积层，输入通道为6，输出通道为16，卷积核大小为5x5，步长为1，无填充
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        # 定义第二个池化层，使用平均池化，池化核大小为2x2，步长为2，无填充
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # 定义第一个全连接层，输入大小为16*6*6，输出大小为120
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        # 定义第一个Dropout层，丢弃概率为0.5
        self.dropout1 = nn.Dropout(0.5)
        # 定义第二个全连接层，输入大小为120，输出大小为84
        self.fc2 = nn.Linear(120, 84)
        # 定义第二个Dropout层，丢弃概率为0.5
        self.dropout2 = nn.Dropout(0.5)
        # 定义第三个全连接层，输入大小为84，输出大小为10
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # 前向传播，依次通过卷积、激活、池化层
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        # 将特征图展开成一维向量
        x = x.view(-1, 16 * 6 * 6)
        # 依次通过全连接层、激活、Dropout层
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        # 输出层
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # 实例化网络并移动到设备上
    net = LeNet().to(device)

    # 定义图像变换，包括灰度化、调整大小、转换为张量和归一化
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 设置训练和测试数据集路径
    train_data_path = 'E:/Python-projects/pytorch/练习/项目实战/mnist_images/train'
    test_data_path = 'E:/Python-projects/pytorch/练习/项目实战/mnist_images/test'

    # 加载训练数据集，使用ImageFolder加载，并应用预处理变换
    trainset = ImageFolder(root=train_data_path, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    # 加载测试数据集，使用ImageFolder加载，并应用预处理变换
    testset = ImageFolder(root=test_data_path, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()
    # 定义优化器为Adam，学习率为0.001
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # 记录每个epoch的训练损失
    train_losses = []

    # 训练网络
    for epoch in range(10):  # 训练10个epoch
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            # 将输入和标签移动到设备上
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
        
        # 计算平均损失并记录
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
    # 在不计算梯度的情况下进行测试
    with torch.no_grad():
        for data in testloader:
            images, labels = data

            # 将输入和标签移动到设备上
            images, labels = images.to(device), labels.to(device)

            # 前向传播
            outputs = net(images)
            # 获取预测结果
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 打印测试集上的准确率
    print(f'Accuracy of the network on the test images: {100 * correct / total}%')

    # 保存模型参数
    PATH = 'E:/Python-projects/pytorch/lenet.pth'
    torch.save(net.state_dict(), PATH)
