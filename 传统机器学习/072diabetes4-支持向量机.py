import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# 初始化并训练SVM模型
svm_clf = SVC(probability=True)

# 使用网格搜索调整超参数
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(estimator=svm_clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

# 输出最佳参数
print(f"Best parameters found: {grid_search.best_params_}")

# 使用最佳参数训练模型
best_svm_clf = grid_search.best_estimator_
best_svm_clf.fit(x_train, y_train)

# 进行预测
y_pred = best_svm_clf.predict(x_test)

# 输出准确率和分类报告
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 计算置信度
y_proba = best_svm_clf.predict_proba(x_test)
confidence = np.max(y_proba, axis=1)
average_confidence = np.mean(confidence)
print(f"Average confidence: {average_confidence:.2f}")
