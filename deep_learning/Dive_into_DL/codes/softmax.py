#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: softmax.py
# @time: 2020/2/13 10:37


import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import sys
# 添加引用模块的地址
sys.path.append(
    "C:\\Users\zhangyu\Documents\my_git\practice\deep_learning\Dive_into_DL")
import d2lzh_pytorch as d2l


path_dataset = \
    'C:\\Users\zhangyu\Documents\my_git\practice\deep_learning\Dive_into_DL\Task01'
mnist_train = torchvision.datasets.FashionMNIST(
    root=path_dataset, train=True,
    download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(
    root=path_dataset, train=False,
    download=True, transform=transforms.ToTensor())

# show result
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 我们可以通过下标来访问任意一个样本
feature, label = mnist_train[0]
print(feature.shape, label)  # Channel x Height x Width

# 如果不做变换输入的数据是图像，我们可以看一下图片的类型参数：
mnist_PIL = torchvision.datasets.FashionMNIST(
    root=path_dataset, train=True, download=True)
PIL_feature, label = mnist_PIL[0]
print(PIL_feature)


# 本函数已保存在d2lzh包中方便以后使用
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(idx)] for idx in labels]


# 以子图形式展示一组图片数据集
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


x = []
y = []
for i in range(10):
    sub_x, sub_y = mnist_train[i]
    x.append(sub_x)
    y.append(sub_y)
show_fashion_mnist(x, get_fashion_mnist_labels(y))

# read dataset
batch_size = 256
train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # print("X size is ", X_exp.size())
    # print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制


def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 获取训练集数据和测试集数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(
    batch_size, root=path_dataset)

# 模型参数初始化
num_inputs = 784    # 28*28
num_outputs = 10

W = torch.tensor(np.random.normal(
    0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 训练模型
num_epochs, lr = 5, 0.1


# 本函数已保存在d2lzh_pytorch包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
          [W, b], lr)

# 模型预测
X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])


# pytorch version
# define net model
num_inputs = 784
num_outputs = 10


class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):  # x 的形状: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))
        return y


# net = LinearNet(num_inputs, num_outputs)

class FlattenLayer(nn.Module):
    # 用于数据size的转换
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x 的形状: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


from collections import OrderedDict

net = nn.Sequential(
    # FlattenLayer(),
    # LinearNet(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()),  # 变换层
        ('linear', nn.Linear(num_inputs, num_outputs))])  # 线性层
    # 或者写成我们自己定义的 LinearNet(num_inputs, num_outputs) 也可以
)

# 初始化模型参数
nn.init.normal_(net.linear.weight, mean=0, std=0.01)
nn.init.constant_(net.linear.bias, val=0)

# 定义损失函数
loss = nn.CrossEntropyLoss()
# 下面是他的函数原型
# class torch.nn.CrossEntropyLoss(weight=None, size_average=None,
# ignore_index=-100, reduce=None, reduction='mean')

# 定义优化函数
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
# 下面是函数原型
# class torch.optim.SGD(params, lr=, momentum=0, dampening=0,
# weight_decay=0, nesterov=False)

# training
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              None, None, optimizer)
