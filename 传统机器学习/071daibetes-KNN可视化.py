import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro

# 启用pandas到R的数据帧转换
pandas2ri.activate()

# 导入所需的R包
importr('ggplot2')
importr('reshape2')

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

# 标准化特征
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

# 将数据集分成两个类别
x_data_0 = x_data[y_data == 0]
y_data_0 = y_data[y_data == 0]
x_data_1 = x_data[y_data == 1]
y_data_1 = y_data[y_data == 1]

# 使用随机过采样来平衡类别
x_data_1_upsampled, y_data_1_upsampled = resample(x_data_1, y_data_1, replace=True, n_samples=len(y_data_0), random_state=42)

# 组合过采样后的数据集
x_data_res = np.vstack((x_data_0, x_data_1_upsampled))
y_data_res = np.hstack((y_data_0, y_data_1_upsampled))

# 特征选择
selector = SelectKBest(f_classif, k=8)
x_data_res = selector.fit_transform(x_data_res, y_data_res)

# 划分数据集为训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x_data_res, y_data_res, test_size=0.2, random_state=42)

# 设置参数网格进行网格搜索
param_grid = {
    'n_neighbors': np.arange(1, 51),
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

# 最佳参数
best_params = grid_search.best_params_
print(f'Best Params: {best_params}')

# 使用最佳参数训练KNN分类器
best_knn = grid_search.best_estimator_

# 使用Bagging方法
bagging_knn = BaggingClassifier(estimator=best_knn, n_estimators=10, random_state=42)
bagging_knn.fit(x_train, y_train)

# 预测训练集和验证集
y_train_pred = bagging_knn.predict(x_train)
y_val_pred = bagging_knn.predict(x_val)

# 计算训练集和验证集的准确率
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')

# 打印分类报告和混淆矩阵
print("\nClassification Report on Validation Set:\n", classification_report(y_val, y_val_pred))
print("\nConfusion Matrix on Validation Set:\n", confusion_matrix(y_val, y_val_pred))

# 可视化训练集和验证集准确率
k_values = param_grid['n_neighbors']
train_accuracies = [accuracy_score(y_train, KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train).predict(x_train)) for k in k_values]
val_accuracies = [accuracy_score(y_val, KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train).predict(x_val)) for k in k_values]

plt.figure(figsize=(12, 6))
plt.plot(k_values, train_accuracies, 'o-', label='Train Accuracy')
plt.plot(k_values, val_accuracies, 'o-', label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.title('Accuracy vs. Number of Neighbors')
plt.legend()
plt.show()

# 绘制混淆矩阵
conf_matrix = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non-Diabetic', 'Diabetic'], rotation=45)
plt.yticks(tick_marks, ['Non-Diabetic', 'Diabetic'])

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.show()

# 预测给定输入数据的疾病概率
def predict_knn(knn_model, input_data, scaler):
    input_data = scaler.transform([input_data])  # 标准化输入数据
    output = knn_model.predict_proba(input_data)
    return output[0][1]  # 返回患病的概率

# 示例输入数据（请修改为实际输入数据）
example_input = [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0]  # 示例数据点

# 预测概率
predicted_prob = predict_knn(bagging_knn, example_input, scaler)
print(f"Predicted probability of having the disease: {predicted_prob:.4f}")

# 根据阈值打印最终诊断（例如，0.5）
threshold = 0.5
if predicted_prob >= threshold:
    print("The model predicts that there is a high probability of having the disease.")
else:
    print("The model predicts that there is a low probability of having the disease.")

# 创建DataFrame用于可视化
data = np.hstack((x_data_res, y_data_res.reshape(-1, 1)))
columns = [f'Feature_{i}' for i in range(x_data_res.shape[1])] + ['Label']
df = pd.DataFrame(data, columns=columns)

# 绘制成对关系图
plt.figure(figsize=(16, 16))
sns.pairplot(df, hue='Label')
plt.savefig('pairplot.png', dpi=300)
plt.show()

# 将数据帧传递给R并创建热图
r_df = pandas2ri.py2rpy(df)

r_code = """
library(ggplot2)
library(reshape2)

# 创建热图
df_melt <- melt(df, id.vars = 'Label')
ggplot(data = df_melt, aes(x = variable, y = value, fill = factor(Label))) +
  geom_tile() +
  scale_fill_manual(values = c("0" = "blue", "1" = "red")) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
"""

# 执行R代码
ro.r.assign('df', r_df)
ro.r(r_code)


# 生成成对关系图（correlates图）
plt.figure(figsize=(16, 16))
sns.pairplot(df)
plt.title('Pairplot (Correlates)')
plt.savefig('pairplot_correlates.png', dpi=300)
plt.show()

# 生成热图
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()
