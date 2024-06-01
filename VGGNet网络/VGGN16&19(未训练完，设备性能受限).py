import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np  # 导入numpy库

# 定义VGGNet类
class VGGNet(nn.Module):
    def __init__(self, architecture, num_classes=10):
        super(VGGNet, self).__init__()
        self.in_channels = 1
        self.conv_layers = self.create_conv_layers(architecture)
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fcs(x)
        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        # 写一个构建层结构的循环来实现数十层的网络结构
        for layer in architecture:
            if type(layer) == int:
                out_channels = layer
                layers += [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(inplace=True)]
                in_channels = out_channels
            elif layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

# Architecture configurations for VGG16 and VGG19
VGG16_architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG19_architecture = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

# 定义数据路径
train_dir = r'E:\Python-projects\pytorch\练习\项目实战\mnist_images\train'
test_dir = r'E:\Python-projects\pytorch\练习\项目实战\mnist_images\test'

# 数据预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载数据
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 实例化模型
vgg16 = VGGNet(VGG16_architecture)
vgg19 = VGGNet(VGG19_architecture)

# 使用GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16.to(device)
vgg19.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer_vgg16 = optim.Adam(vgg16.parameters(), lr=0.001)
optimizer_vgg19 = optim.Adam(vgg19.parameters(), lr=0.001)

# 训练和验证函数
def train_model(model, optimizer, criterion, train_loader, test_loader, num_epochs=10):
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(test_loader.dataset)
        val_acc = val_corrects.double() / len(test_loader.dataset)

        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)

        print(f'Epoch {epoch}/{num_epochs - 1}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return train_loss_history,
    train_acc_history, val_loss_history, val_acc_history

# 训练VGG16模型
print("Training VGG16")
vgg16_train_loss, vgg16_train_acc, vgg16_val_loss, vgg16_val_acc = train_model(vgg16, optimizer_vgg16, criterion, train_loader, test_loader, num_epochs=10)
torch.save(vgg16.state_dict(), "vgg16_mnist.pth")

# 训练VGG19模型
print("Training VGG19")
vgg19_train_loss, vgg19_train_acc, vgg19_val_loss, vgg19_val_acc = train_model(vgg19, optimizer_vgg19, criterion, train_loader, test_loader, num_epochs=10)
torch.save(vgg19.state_dict(), "vgg19_mnist.pth")

# 可视化损失率和准确率
def plot_metrics(train_loss, train_acc, val_loss, val_acc, model_name):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'bo-', label='Training loss')
    plt.plot(epochs, val_loss, 'ro-', label='Validation loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'bo-', label='Training accuracy')
    plt.plot(epochs, val_acc, 'ro-', label='Validation accuracy')
    plt.title(f'{model_name} Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

# 可视化VGG16的训练结果
plot_metrics(vgg16_train_loss, vgg16_train_acc, vgg16_val_loss, vgg16_val_acc, "VGG16")

# 可视化VGG19的训练结果
plot_metrics(vgg19_train_loss, vgg19_train_acc, vgg19_val_loss, vgg19_val_acc, "VGG19")

