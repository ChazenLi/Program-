## specification of approach the linear 
## 导入必要的库：例如 NumPy 用于数值计算，Matplotlib 用于绘图。
## 数据定义：定义输入数据和目标输出数据。
## 前向计算函数：定义一个函数来计算模型的预测值。
## 损失函数：定义一个函数来计算预测值与真实值之间的误差。
## 参数搜索或优化方法：使用一个方法（如穷举法、梯度下降等）来搜索或优化模型参数，使损失最小化。
## 绘图或结果展示：将结果可视化，便于理解和分析。


import numpy as np
import matplotlib.pyplot as plt
 
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
 
 
def forward(x):
    return x*w
 
 
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y)**2
 
 
# 穷举法
w_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    print("w=", w)
    l_sum = 0
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)
        loss_val = loss(x_val, y_val)
        l_sum += loss_val
        print('\t', x_val, y_val, y_pred_val, loss_val)
    print('MSE=', l_sum/3)
    w_list.append(w)
    mse_list.append(l_sum/3)
    
plt.plot(w_list,mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()   