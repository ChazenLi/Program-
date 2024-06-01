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
class LinearRegressionModel(nn.Module):  # Corrected the class name and parent class
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # Corrected the attribute initialization

    def forward(self, x):
        return self.linear(x)
    
## instantiate the model
model = LinearRegressionModel()

## define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

## main part of training the model
num_epochs = 10000  # Corrected the variable name
for epoch in range(num_epochs):  # Use range() to iterate over the number of epochs
    model.train()

    ## forward pass
    y_pred = model(x_data)

    ## loss calculation
    loss = criterion(y_pred, y_data)

    ## backward pass
    optimizer.zero_grad()  # Clear the gradients
    loss.backward()  # Compute the gradients
    optimizer.step()  # Update the parameters

    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

## visualization of the result
model.eval()
with torch.no_grad():
    predicted = model(x_data).detach().numpy()
    plt.plot(x_data.numpy(), y_data.numpy(), 'ro', label='Original data')
    plt.plot(x_data.numpy(), predicted, label='Fitted line')
    plt.legend()
    plt.show()  # Corrected the show() method

    
