import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 读取数据
data = pd.read_csv('E:\Python-projects\pytorch\练习\diabetes.csv')

# 特征和标签
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# K折交叉验证
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
reports = []

for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 模型构建
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # 训练模型
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0, validation_split=0.2)
    
    # 模型评估
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    report = classification_report(y_test, y_pred, output_dict=True)
    reports.append(report)

# 平均准确率
average_accuracy = np.mean(accuracies)
print(f'Average Accuracy: {average_accuracy}')

# 打印分类报告
for i, report in enumerate(reports):
    print(f'Fold {i+1} Classification Report:')
    print(pd.DataFrame(report))

# 平均置信度
average_confidence = np.mean([report['weighted avg']['precision'] for report in reports])
print(f'Average Confidence: {average_confidence}')
