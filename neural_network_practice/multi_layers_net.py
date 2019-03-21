#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: multi_layers_net.py
# @time: 2019/1/16 15:02

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    delta = 1e-7
    cee = -np.sum(t * np.log(y + delta))
    return cee


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 还原值
        it.iternext()

    return grad


class MultiLayerNet:

    def __init__(self, num_layers, vector_size, weight_init_std=0.01):
        # initialize weights
        if num_layers < 1:
            print("Number of layers should be more than zero!")
            return

        self.num_layers = num_layers
        self.params = {}
        for i in range(num_layers):
            self.params['w' + str(i)] = \
                weight_init_std * \
                np.random.randn(vector_size[i], vector_size[i+1])
            self.params['b' + str(i)] = np.zeros(vector_size[i+1])

    def predict(self, x):
        z_old = x
        for i in range(self.num_layers):
            w = self.params['w' + str(i)]
            b = self.params['b' + str(i)]
            a = np.dot(z_old, w) + b
            if i == self.num_layers:
                z = softmax(a)
            else:
                z = sigmoid(a)
                z_old = z

        return z

    def loss(self, x, t):
        y = self.predict(x)
        cee = cross_entropy_error(y, t)
        return cee

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        grads = {}
        for i in range(self.num_layers):
            grads['w' + str(i)] = \
                numerical_gradient(loss_w, self.params['w' + str(i)])
            grads['b' + str(i)] = \
                numerical_gradient(loss_w, self.params['b' + str(i)])

        return grads
