'''
XOR by Machine Learning on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python xor.py
# Prerequisite: tensorflow (see tensorflow.org)


import tensorflow as tf
import numpy as np

learning_rate = 0.1
x_data = np.array([0., 0., 0., 1., 1., 0., 1., 1.])
x_data = np.reshape(x_data,[4,1,2])
xor_out = np.array([0., 1., 1., 1.])
y_data = xor_out
y_data = np.reshape(y_data,[4,1,1])
n = y_data.shape[0]
x = tf.placeholder(tf.float32,shape=(1,2))
y = tf.placeholder(tf.float32,shape=(1,1))

W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
b = tf.Variable(tf.ones([1, 1]))

yp = tf.nn.softmax(tf.matmul(x, W) + b)
# loss = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(yp, y))
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(yp)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# loss = tf.reduce_mean(tf.square(yp - y))
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(100):
        offset = i%n
        feed_dict = {x:x_data[offset,:], y:y_data[offset,:]}
        _, l, yp_, y_, x_ = session.run([train_step, loss, yp, y, x], feed_dict=feed_dict)
        #if (i+1) % 100 == 0:
        print("--- %d: %lf" % (i, l))
        print(yp_)
        print(y_)
        print(x_)
        print(session.run(W))
        print(session.run(b))
    x_ = tf.constant(np.reshape(x_data, [4, 2]), dtype=tf.float32)
    yp_ = tf.matmul(x_, W) + b
    print(x_.eval())
    print(yp_.eval())