import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 从 Excel 文件中读取数据
data = pd.read_excel('E:/training data/20240111.xlsx')

# 计算标准差和均值
std_dev = data.std().values
mean = data.mean().values

print("标准差：")
print(std_dev)
print("\n均值：")
print(mean)

# 使用均值和标准差作为双高斯函数的初始参数
A = np.ones((2, 19))  # 初始化为全1，2行19列的矩阵
mu = np.tile(mean.reshape(1, -1), (2, 1))  # 将均值扩展为2行的矩阵
sigma = np.tile(std_dev.reshape(1, -1), (2, 1))  # 将标准差扩展为2行的矩阵
learning_rate = 0.001
num_iterations = 1000

# 提取荧光强度数据
fluorescence_data = data.values[:, 1:]  # 排除第一列时间序列数据

# 定义双高斯函数
def double_gaussian(A, mu, sigma):
    gauss1 = A[0, :] * np.exp(-((fluorescence_data - mu[0, :])**2) / (2 * sigma[0, :]**2))
    gauss2 = A[1, :] * np.exp(-((fluorescence_data - mu[1, :])**2) / (2 * sigma[1, :]**2))
    return gauss1 + gauss2

# 定义损失函数（均方误差）
def compute_loss(A, mu, sigma):
    y_pred = double_gaussian(A, mu, sigma)
    return np.mean((y_pred - fluorescence_data) ** 2)

# 计算梯度
def compute_gradients(A, mu, sigma):
    y_pred = double_gaussian(A, mu, sigma)
    
    # 使用矩阵计算
    m = len(fluorescence_data)
    X1 = np.exp(-((fluorescence_data - mu[0, :]) ** 2) / (2 * sigma[0, :] ** 2)).reshape(m, 1)
    X2 = np.exp(-((fluorescence_data - mu[1, :]) ** 2) / (2 * sigma[1, :] ** 2)).reshape(m, 1)
    
    dA1 = np.mean(2 * np.matmul((y_pred - fluorescence_data), X1))
    dmu1 = np.mean(2 * np.matmul((y_pred - fluorescence_data) * A[0, :] * (fluorescence_data - mu[0, :]), X1) / (sigma[0, :] ** 2))
    dsigma1 = np.mean(2 * np.matmul((y_pred - fluorescence_data) * A[0, :] * ((fluorescence_data - mu[0, :]) ** 2), X1) / (sigma[0, :] ** 3))
    
    dA2 = np.mean(2 * np.matmul((y_pred - fluorescence_data), X2))
    dmu2 = np.mean(2 * np.matmul((y_pred - fluorescence_data) * A[1, :] * (fluorescence_data - mu[1, :]), X2) / (sigma[1, :] ** 2))
    dsigma2 = np.mean(2 * np.matmul((y_pred - fluorescence_data) * A[1, :] * ((fluorescence_data - mu[1, :]) ** 2), X2) / (sigma[1, :] ** 3))
    
    return dA1, dmu1, dsigma1, dA2, dmu2, dsigma2

# 梯度下降法
losses = []
for i in range(num_iterations):
    dA1, dmu1, dsigma1, dA2, dmu2, dsigma2 = compute_gradients(A, mu, sigma)
    
    A[0, :] -= learning_rate * dA1
    mu[0, :] -= learning_rate * dmu1
    sigma[0, :] -= learning_rate * dsigma1
    
    A[1, :] -= learning_rate * dA2
    mu[1, :] -= learning_rate * dmu2
    sigma[1, :] -= learning_rate * dsigma2
    
    loss = compute_loss(A, mu, sigma)
    losses.append(loss)
    if i % 100 == 0:
        print(f"Iteration {i}: loss={loss:.4f}")

# 绘制损失变化曲线
plt.plot(range(num_iterations), losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over iterations')
plt.show()

# 绘制拟合结果
plt.figure(figsize=(10, 6))
for i in range(19):
    plt.scatter(fluorescence_data[:, i], color='red', label='Data')
    x_fit = np.linspace(np.min(fluorescence_data), np.max(fluorescence_data), 100)
    y_fit = double_gaussian(A, mu, sigma)[0, :]
    plt.plot(x_fit, y_fit, label=f'Column {i+1}')
plt.xlabel('Fluorescence')
plt.ylabel('Density')
plt.title('Double Gaussian Fit')
plt.legend()
plt.show()
