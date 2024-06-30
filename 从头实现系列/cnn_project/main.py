import numpy as np
from model import Model
from net import Net
from layers.dense import Dense
from layers.activation import ReLU
from loss.cross_entropy import CrossEntropyLoss
from optimizers.adam import Adam

# 定义网络
layer1 = Dense(784, 128)
layer2 = ReLU()
layer3 = Dense(128, 10)
net = Net([layer1, layer2, layer3])

# 定义损失函数和优化器
loss_fn = CrossEntropyLoss()
optimizer = Adam(lr=0.001)

# 定义模型
model = Model(net, loss_fn, optimizer)

# 训练
train_X = np.random.rand(100, 784)  # 示例训练数据
train_Y = np.zeros((100, 10))  # 示例训练标签
pred = model.forward(train_X)
loss, grads = model.backward(pred, train_Y)
model.apply_grad(grads)

# 推断
test_X = np.random.rand(10, 784)  # 示例测试数据
test_pred = model.forward(test_X)
print(test_pred)
