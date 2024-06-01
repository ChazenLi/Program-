import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

# 划分数据集为训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# 设置K值的范围
k_values = np.arange(1, 38)
train_accuracies = []
val_accuracies = []

# 计算每个K值的训练集和验证集准确率
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_train_pred = knn.predict(x_train)
    y_val_pred = knn.predict(x_val)
    train_accuracies.append(accuracy_score(y_train, y_train_pred))
    val_accuracies.append(accuracy_score(y_val, y_val_pred))

# 选择最佳K值
best_k = k_values[np.argmax(val_accuracies)]
print(f'Best K value: {best_k}')

# 使用最佳K值训练KNN分类器
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)

# 预测训练集和验证集
y_train_pred = knn.predict(x_train)
y_val_pred = knn.predict(x_val)

# 计算训练集和验证集的准确率
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f'Train Accuracy: {train_accuracy:.4f}')
print(f'Validation Accuracy: {val_accuracy:.4f}')

# 打印分类报告和混淆矩阵
print("\nClassification Report on Validation Set:\n", classification_report(y_val, y_val_pred))
print("\nConfusion Matrix on Validation Set:\n", confusion_matrix(y_val, y_val_pred))

# 可视化训练集和验证集准确率
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
predicted_prob = predict_knn(knn, example_input, scaler)
print(f"Predicted probability of having the disease: {predicted_prob:.4f}")

# 根据阈值打印最终诊断（例如，0.5）
threshold = 0.5
if predicted_prob >= threshold:
    print("The model predicts that there is a high probability of having the disease.")
else:
    print("The model predicts that there is a low probability of having the disease.")

