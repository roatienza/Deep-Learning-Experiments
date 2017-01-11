'''
Linear Regression using Pseudo Inverse on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python distributions.py
# Prerequisite: tensorflow (see tensorflow.org)
#             : matplotlib (http://matplotlib.org/)

import tensorflow as tf
import numpy as np

x_data = np.random.rand(4000).astype("float32")
y_data = 2.0*x_data*x_data - x_data*3.0 - 12.0
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
Z = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

y = Z*x_data*x_data + W*x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.4)
train_step = optimizer.minimize(loss)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(10000):
        session.run(train_step)
        if ((i+1) % 500) == 0:
            print(i, session.run(Z), session.run(W), session.run(b))
    print(session.run(Z), session.run(W), session.run(b))
    # for i in xrange(4):
    # print(i, x_data[i % 4], session.run(y, feed_dict={x: x_data[i % 4]}))
