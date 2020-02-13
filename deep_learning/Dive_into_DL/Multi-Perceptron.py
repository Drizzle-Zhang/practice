#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: Multi-Perceptron.py
# @time: 2020/2/13 19:28

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(
    "C:\\Users\zhangyu\Documents\my_git\practice\deep_learning\Dive_into_DL")
import d2lzh_pytorch as d2l
print(torch.__version__)


# display Relu
def xyplot(x_vals, y_vals, name):
    # d2l.set_figsize(figsize=(5, 2.5))
    plt.plot(x_vals.detach().numpy(), y_vals.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = x.relu()
xyplot(x, y, 'relu')
y.sum().backward()
xyplot(x, x.grad, 'grad of relu')

# sigmoid
y = x.sigmoid()
xyplot(x, y, 'sigmoid')
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of sigmoid')

# tanh
y = x.tanh()
xyplot(x, y, 'tanh')
x.grad.zero_()
y.sum().backward()
xyplot(x, x.grad, 'grad of tanh')

# 多层感知机从零开始的实现
# 获取训练集数据和测试集数据
batch_size = 256
path_dataset = \
    'C:\\Users\zhangyu\Documents\my_git\practice\deep_learning\Dive_into_DL\Task01'
train_iter, test_iter = d2l.load_data_fashion_mnist(
    batch_size, root=path_dataset)

# 定义模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(
    np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(
    np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)


# 定义激活函数
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))


# 定义网络
def net(X):
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)
    return torch.matmul(H, W2) + b2


# 定义损失函数
loss = torch.nn.CrossEntropyLoss()

# training
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)

# 多层感知机pytorch实现
# init
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = torch.nn.Sequential(
    d2l.FlattenLayer(),
    torch.nn.Linear(num_inputs, num_hiddens),
    torch.nn.ReLU(),
    torch.nn.Linear(num_hiddens, num_outputs),
)

for params in net.parameters():
    torch.nn.init.normal_(params, mean=0, std=0.01)

# training
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,
                                                    root=path_dataset)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, optimizer)
