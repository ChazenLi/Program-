import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

## 自定义 Dataset 类
class DiabetesDataset(Dataset):
    def __init__(self, file_path):
        data = pd.read_csv(file_path)
        self.X = data.iloc[:, :-1].values  # 特征
        self.y = data.iloc[:, -1].values   # 标签
        
        # 标准化特征
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

## 数据可视化
def visualize_data(data):
    with pd.option_context('mode.use_inf_as_na', True):
        df = pd.read_csv(data)
    sns.pairplot(df, diag_kind='kde')
    plt.show()

## 加载数据
file_path = 'E:/Python-projects/pytorch/练习/diabetes.csv'
train_dataset = DiabetesDataset(file_path)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

## 定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(train_dataset.X.shape[1], 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 2)  # 假设输出类别数为2，根据实际情况调整
        self.dropout = torch.nn.Dropout(p=0.5)  # Dropout层
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.dropout(x)
        x = F.relu(self.l2(x))
        x = self.dropout(x)
        x = F.relu(self.l3(x))
        x = self.dropout(x)
        x = F.relu(self.l4(x))
        x = self.dropout(x)
        return self.l5(x)

model = Net()

## 构建损失函数和优化器，加入 L2 正则化
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.001)  # 加入L2正则化项

## 记录损失和准确率
train_losses = []
test_accuracies = []
test_losses = []

## 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0
    train_losses.append(running_loss / len(train_loader))

## 测试函数
def test():
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    test_losses.append(test_loss / len(test_loader))
    print('Accuracy on test set: %d %%' % accuracy)

## 可视化训练损失和测试准确率
def plot_metrics():
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'b', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, test_losses, 'b', label='Test loss')
    plt.title('Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, test_accuracies, 'b', label='Test accuracy')
    plt.title('Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()

## 任意输入数据的预测
def predict(input_data):
    model.eval()
    with torch.no_grad():
        input_data = torch.tensor(input_data, dtype=torch.float32)
        outputs = model(input_data)
        _, predicted = torch.max(outputs.data, 1)
    return predicted.item()

if __name__ == '__main__':
    visualize_data(file_path)
    for epoch in range(100):
        train(epoch)
        test()
    plot_metrics()

    # 预测示例
    sample_data = [[0.5, -0.2, 0.1, 0.4, 0.6, 0.3, 0.8, -0.5]]  # 示例数据，根据实际数据调整
    prediction = predict(sample_data)
    print(f'Predicted class for input data: {prediction}')
