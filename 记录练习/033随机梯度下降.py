import matplotlib.pyplot as plt

#initiate data
x_data = [0.46, 2.03, 3.97, 4.71, 8.03, 11.14, 15.44, 15.98, 16.73, 18.51]
y_data = [3.27, 6.15, 8.13, 9.69, 12.87, 13.0, 17.6, 18.08, 18.18, 19.62]

# initiate parameters define
w = 1.0

# forward function
def forward(x):
    return x * w

# loss function:
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

# gradient function
def gradient(x, y):
    return 2*x*(x*w - y)

epoch_list = []
loss_list = []
print('predict (before training)', 4, forward(4))
for epoch in range(100):
    for x, y in zip(x_data, y_data):
        grad = gradient(x, y)
        w = w - 0.01*grad
        print("\tgrad:", x, y, grad)
        l = loss(x, y)
    print("progress:", epoch, "w=", w, "loss=", l)
    epoch_list.append(epoch)
    loss_list.append(l)

print('predict (before training)', 4, forward(4))
plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show

