import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

## 生成二分类数据
X, y = make_classification(n_samples=1000, 
                           n_features=10, 
                           n_classes=2,
                           random_state= 42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,random_state= 42
)

## transfor to tensor:
X_train = torch.tensor(X_train, dtype= torch.float32)
X_test = torch.tensor(X_test, dtype= torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

## define the logistic model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

## initiate the model:
model = LogisticRegressionModel(input_dim=10)

## define the loss funciton and optmizer:
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

## train the model:
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()

    ## forward broadcast
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    ## backward broadcast and optimize:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}],
              Loss: {loss.item():.4f}')

## evaluate the model:
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_classes = predictions.round()
    accuracy = (predicted_classes.eq(y_test).sum / y_test.shape[0]).item
    print(f'Accuracy: {accuracy:.4f}')
