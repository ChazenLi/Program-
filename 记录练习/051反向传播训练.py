## import the packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

## prepare the data 
x_data = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

## model define 
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
    
## instantiate the model
model = LinearRegressionModel()

## define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

## lists to store losses and parameters for visualization
losses = []
w_values = []
b_values = []

## main part of training the model
num_epochs = 100000
for epoch in range(num_epochs):
    model.train()

    ## forward pass
    y_pred = model(x_data)

    ## loss calculation
    loss = criterion(y_pred, y_data)

    ## backward pass
    optimizer.zero_grad()  # Clear the gradients
    loss.backward()  # Compute the gradients
    optimizer.step()  # Update the parameters

    ## store the loss and parameters
    losses.append(loss.item())
    w_values.append(model.linear.weight.item())
    b_values.append(model.linear.bias.item())

    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

## visualization of the result
model.eval()
with torch.no_grad():
    predicted = model(x_data).detach().numpy()
    plt.plot(x_data.numpy(), y_data.numpy(), 'ro', label='Original data')
    plt.plot(x_data.numpy(), predicted, label='Fitted line')
    plt.legend()
    plt.show()

## visualize the loss over epochs
plt.figure()
plt.plot(range(num_epochs), losses, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()
plt.show()

## visualize the parameter updates
plt.figure()
plt.plot(range(num_epochs), w_values, label='Weight')
plt.plot(range(num_epochs), b_values, label='Bias')
plt.xlabel('Epoch')
plt.ylabel('Parameter value')
plt.title('Parameter updates over Epochs')
plt.legend()
plt.show()
