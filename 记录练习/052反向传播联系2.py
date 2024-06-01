import numpy as np
import matplotlib.pyplot as plt
import torch

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [3.0, 6.0, 9.0, 12.0]

# Initiate the parameters
w1 = torch.tensor([1.0], requires_grad=True)
w2 = torch.tensor([1.0], requires_grad=True)
w3 = torch.tensor([1.0], requires_grad=True)

# Forward function define
def forward(x):
    return w1 * x * x + w2 * x + w3

# Loss function define
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print('Predict (before training)', 4, forward(4).item())

learning_rate = 0.001  # Reduced learning rate

for epoch in range(10000):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()

        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_([w1, w2, w3], max_norm=1.0)

        print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), w3.grad.item())

        w1.data = w1.data - learning_rate * w1.grad.data
        w2.data = w2.data - learning_rate * w2.grad.data
        w3.data = w3.data - learning_rate * w3.grad.data

        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()

    print('Epoch:', epoch, l.item())

print('Predict (after training)', 4, forward(4).item())


## 之所以最后会出现预测的结果和普适性的结果不一样的区别是：
## 我们采用的模型，不一定适合本身数据的模式。
