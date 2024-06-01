## 实现线性模型（y = w*x +b）
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 数据定义
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([5.0, 8.0, 11.0])

# 参数网格
W = np.arange(0.0, 4.1, 0.1)
B = np.arange(0.0, 4.1, 0.1)
w, b = np.meshgrid(W, B)

# 前向计算函数
def forward(x, w, b):
    return x * w + b

# 损失函数
def loss(y_pred, y):
    return (y_pred - y) ** 2

# 计算损失
mse_list = np.zeros_like(w)
for x_val, y_val in zip(x_data, y_data):
    y_pred = forward(x_val, w, b)
    mse_list += loss(y_pred, y_val)

# 取均值
mse_list /= len(x_data)

# 绘制3D图表
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(w, b, mse_list, cmap='viridis')

ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('Mean Squared Error (MSE)')
ax.set_title('MSE over weight and bias')

plt.show()




