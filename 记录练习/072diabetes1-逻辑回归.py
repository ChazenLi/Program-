import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 检查当前工作目录
print(f"Current working directory: {os.getcwd()}")

# 确保文件路径正确
file_path = 'E:/Python-projects/pytorch/练习/diabetes.csv'
if not os.path.isfile(file_path):
    raise FileNotFoundError(f"{file_path} not found in the current directory.")

# 准备数据集
xy = np.genfromtxt(file_path, delimiter=',', dtype=np.float32, skip_header=1)
x_data = xy[:, :-1]
y_data = xy[:, -1]

# 数据标准化
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# 拆分数据集为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# 初始化并训练逻辑回归模型，增加正则化
log_reg = LogisticRegression(max_iter=2000, penalty='l2', C=1.0)  # 使用L2正则化
log_reg.fit(x_train, y_train)

# 进行预测
y_pred = log_reg.predict(x_test)

# 输出准确率和分类报告
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 初始化CCA，n_components设置为特征数量的最小值
n_components = min(x_data.shape[1], 1)  # 这里1是标签的维度
cca = CCA(n_components=n_components)
y_data_reshaped = y_data.reshape(-1, 1)
x_c, y_c = cca.fit_transform(x_data, y_data_reshaped)

# 绘制CCA关系图
plt.figure(figsize=(10, 6))
plt.scatter(x_c[:, 0], y_c, c=y_data, cmap='viridis')
plt.xlabel('Component 1')
plt.ylabel('Target')
plt.title('CCA Relationship Plot')
plt.colorbar()
plt.show()

# 使用PCA来模拟RDA
pca = PCA(n_components=2)
x_r = pca.fit_transform(x_data)

# 绘制RDA关系图
plt.figure(figsize=(10, 6))
plt.scatter(x_r[:, 0], x_r[:, 1], c=y_data, cmap='viridis')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('RDA Relationship Plot')
plt.colorbar()
plt.show()

# 将数据转换为DataFrame
df = pd.DataFrame(x_data)
df['target'] = y_data

# 计算相关矩阵
corr = df.corr()

# 绘制热图
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
