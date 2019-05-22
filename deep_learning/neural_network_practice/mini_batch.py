#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: mini_batch.py
# @time: 2019/1/17 15:15

import numpy as np
from dataset.mnist import load_mnist
from multi_layers_net import MultiLayerNet
from time import time


def train_test(iters_num, batch_size, learning_rate):
    (x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    train_size = x_train.shape[0]
    iter_per_epoch = max(train_size/batch_size, 1)

    net = MultiLayerNet(num_layers=2, vector_size=[784, 50, 10])

    for i in range(iters_num):
        # get mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # calculate accuracy
        grad = net.numerical_gradient(x_batch, t_batch)

        # update parameters
        keys = ['w' + str(i) for i in range(net.num_layers)]
        keys.extend(['b' + str(i) for i in range(net.num_layers)])
        for key in set(keys):
            net.params[key] -= learning_rate * grad[key]

        # calculate loss
        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        # calculate accuracy of each epoch
        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, t_train)
            test_acc = net.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(str(i // iter_per_epoch) + "th epoch  train acc, test acc | "
                  + str(train_acc) + ", " + str(test_acc))

    return


if __name__ == '__main__':
    start_time = time()
    # hyperparameters
    iters_num0 = 10000
    batch_size0 = 100
    learning_rate0 = 0.1
    train_test(iters_num0, batch_size0, learning_rate0)
    end_time = time()
