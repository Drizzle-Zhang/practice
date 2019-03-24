#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: simple_demos.py
# @time: 3/21/19 9:36 PM

import tensorflow as tf
import numpy as np
import random

"""
# computation graph
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable(
        "v", shape=[1], initializer=tf.zeros_initializer
        # define variable v and initialize
    )

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope('', reuse=True):
        print(sess.run(tf.get_variable('v')))


# tensor
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = tf.add(a, b, name="add")
print(result)  # Tensor("add:0", shape=(2,), dtype=float32)
"""


"""train a simple neural network"""
batch_size = 8

# parameters of NN
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# input
x = tf.placeholder(tf.float32, shape=(None, 2), name='x')
y_label = tf.placeholder(tf.float32, shape=(None, 1), name='y_label')

# forward-propagation process
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# loss function and back-propagation
cross_entropy = \
    -tf.reduce_mean(y_label * tf.log(tf.clip_by_value(y, 1e-10, 1)) +
                    (1 - y_label) * tf.log(tf.clip_by_value(1-y, 1e-10, 1)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# generate simulated data
rdm = np.random.RandomState(1)
dataset_size = 128
x_input = rdm.rand(dataset_size, 2)
y_input = [[int(x1+x2 < 1)] for (x1, x2) in x_input]

# create a session
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print(sess.run(w1))
    print(sess.run(w2))

    steps = 5000  # num of iteration
    for i in range(steps):
        idx_batch = random.sample(range(dataset_size), batch_size)

        # update parameter
        sess.run(train_step,
                 feed_dict={x: [x_input[idx, :] for idx in idx_batch],
                            y_label: [y_input[idx] for idx in idx_batch]})

        # print cross entropy of all data per 1000 steps
        if i % 1000 == 0:
            total_cross_entropy = sess.run(
                cross_entropy, feed_dict={x: x_input, y_label: y_input}
            )
            print("After %d training steps, cross entropy on all data is %f" %
                  (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))

