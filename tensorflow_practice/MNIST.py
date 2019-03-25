#!/usr/bin/env python
# _*_ coding: utf-8 _*_

# @author: Drizzle_Zhang
# @file: MNIST.py
# @time: 3/24/19 4:48 PM

from time import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    """
    a auxiliary function calculating forward-propagation results based on given
    parameters
    """
    if avg_class is None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) + biases2
    else:
        layer1 = \
            tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) +
                       avg_class.average(biases1))
        return tf.matmul(
            layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32, shape=(None, input_node), name='x')
    y_label = tf.placeholder(tf.float32, shape=(None, output_node),
                             name='y_label')

    # generate parameters
    weights1 = tf.Variable(
        tf.truncated_normal([input_node, layer1_node], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[layer1_node]))
    weights2 = tf.Variable(
        tf.truncated_normal([layer1_node, output_node], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[output_node]))

    # forward-propagation without moving average
    y = inference(x, None, weights1, biases1, weights2, biases2)

    global_step = tf.Variable(0, trainable=False)

    # initialize moving average class
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step)

    # use moving-average method for trainable parameters
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # forward-propagation with moving average
    average_y = inference(x, variable_averages, weights1, biases1,
                          weights2, biases2)

    # loss function
    # cross entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_label, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # L2 regularization
    regularizer = tf.contrib.layers.l2_regularizer(regularization_rate)
    regularization = regularizer(weights1) + regularizer(weights2)
    loss = cross_entropy_mean + regularization

    # exponential decay learning rate
    learning_rate = tf.train.exponential_decay(
        learning_rate_base, global_step, mnist.train.num_examples/batch_size,
        learning_rate_decay)

    # optimize loss function
    train_step = tf.train.GradientDescentOptimizer(
        learning_rate).minimize(loss, global_step=global_step)

    # use group method to update two groups of parameters
    train_op = tf.group(train_step, variables_averages_op)

    # calculate accuracy
    correct_prediction = tf.equal(tf.argmax(average_y, 1),
                                  tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # validation data
        validate_feed = {x: mnist.validation.images,
                         y_label: mnist.validation.labels}
        # test data
        test_feed = {x: mnist.test.images, y_label: mnist.test.labels}

        # train neural network
        for i in range(training_steps):
            # print validating results per 1000 steps
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy using "
                      "average model is %g" % (i, validate_acc))
            # train in a batch
            xs, ys = mnist.train.next_batch(batch_size)
            sess.run(train_op, feed_dict={x: xs, y_label: ys})

        # print testing results finally
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, test accuracy using average "
              "model is %g" % (training_steps, test_acc))

    return


if __name__ == '__main__':
    time_start = time()
    # download data
    mnist_data = input_data.read_data_sets("./MNIST_data", one_hot=True)

    # constants correspond to dataset
    input_node = 784
    output_node = 10

    # hyper-parameter
    layer1_node = 500  # node num of hidden layer
    batch_size = 100
    learning_rate_base = 0.8
    learning_rate_decay = 0.99
    regularization_rate = 0.0001
    training_steps = 30000
    moving_average_decay = 0.99

    # run training function
    train(mnist_data)

    time_end = time()
    print(time_end - time_start)
