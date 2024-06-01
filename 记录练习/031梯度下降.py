import numpy as np
import matplotlib.pyplot as plt

# 数据
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([1.9, 4.2, 6.6])

# 初始化参数
w = 2.0
b = 0.0
learning_rate = 0.01
num_iterations = 1000

# 定义前向计算函数
def forward(x):
    return w * x + b

# 定义损失函数（均方误差）
def compute_loss(x, y):
    y_pred = forward(x)
    return np.mean((y_pred - y) ** 2)

# 定义梯度计算
def compute_gradients(x, y):
    y_pred = forward(x)
    grad_w = np.mean(2 * (y_pred - y) * x)
    grad_b = np.mean(2 * (y_pred - y))
    return grad_w, grad_b

# 梯度下降法
losses = []
for i in range(num_iterations):
    grad_w, grad_b = compute_gradients(x_data, y_data)
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b
    loss = compute_loss(x_data, y_data)
    losses.append(loss)
    if i % 100 == 0:
        print(f"Iteration {i}: w={w:.4f}, b={b:.4f}, loss={loss:.4f}")

# 绘制损失变化曲线
plt.plot(range(num_iterations), losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over iterations')
plt.show()

# 绘制拟合结果
plt.scatter(x_data, y_data, color='red')
plt.plot(x_data, forward(x_data), color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear regression result')
plt.show()
