## 在处理多维特征的输入时，我们需要把每一个特征赋予相对应的权重；
## 在进行逻辑回归的时候把每一个维度的x乘以对应的权值；
## 并加上一个偏执量，使用sigma函数进行二分类
## DIabetes 数据库已经有了类似的标定注释模式，可以对此进行学习和借鉴

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Check the current working directory
print(f"Current working directory: {os.getcwd()}")

# Ensure the file path is correct
file_path = 'E:/Python-projects/pytorch/练习/diabetes.csv'
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"{file_path} not found in the current directory.")

# Prepare dataset
xy = np.genfromtxt(file_path, delimiter=',', dtype=np.float32, skip_header=1)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])

# Design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

# Construct loss and optimizer
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

epoch_list = []
loss_list = []
nums_epoch = 10000

# Training cycle: forward, backward, update
for epoch in range(nums_epoch):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{nums_epoch}], Loss: {loss.item():.4f}')
    epoch_list.append(epoch)
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plot the loss curve
plt.plot(epoch_list, loss_list)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Curve')
plt.show()

# Function to predict the probability of having the disease
def predict(model, input_data):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor)
        return output.item()

# Example input data (modify this with actual input data)
example_input = [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]  # Example data point

# Normalize the input data if necessary (optional, depending on your dataset preprocessing)
example_input = np.array(example_input).reshape(1, -1)  # Reshape to (1, 8)

# Predict the probability
predicted_prob = predict(model, example_input)
print(f"Predicted probability of having the disease: {predicted_prob:.4f}")

# Print the final diagnosis based on a threshold (e.g., 0.5)
threshold = 0.5
if predicted_prob >= threshold:
    print("The model predicts that there is a high probability of having the disease.")
else:
    print("The model predicts that there is a low probability of having the disease.")
