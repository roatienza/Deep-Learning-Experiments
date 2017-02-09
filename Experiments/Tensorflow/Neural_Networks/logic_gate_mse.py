'''
Logical Operation by 2-layer Neural Networks on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python logic_gate_mse.py
# Prerequisite: tensorflow (see tensorflow.org)


from __future__ import print_function

import tensorflow as tf
import numpy as np

learning_rate = 0.6
x_data = np.reshape(np.array( [[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32 ),[4,2])
# try other logics; xor = [0., 1., 1., 0.], or = [0., 1., 1., 1.], and = [0., 0., 0., 1.], etc
logic_out = np.array([0., 1., 1., 0.], dtype=np.float32)
y_data = np.reshape(logic_out,[4,1])
# n = y_data.shape[0]

x = tf.placeholder(tf.float32, shape=(None, 2))
y = tf.placeholder(tf.float32, shape=(None, 1))

# try other values for nhidden
nhidden = 32
W0 = tf.Variable(tf.random_normal([2, nhidden],stddev=0.1))
b0 = tf.Variable(tf.zeros([nhidden]))
# b0 = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[nhidden]))


W1 = tf.Variable(tf.random_normal([nhidden, 1],stddev=0.1))
b1 = tf.Variable(tf.zeros([1]))
# b1 = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=[1]))

hidden = tf.matmul(x, W0) + b0
yp = tf.matmul(tf.nn.relu(hidden), W1) + b1
# yp = tf.matmul(hidden, W1) + b1

loss = tf.reduce_mean(tf.square(yp - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(1000):
        # mini-batch can also be used but we have a small set of data only
        # offset = (i*2)%(n-2)
        # feed_dict ={x:x_data[offset:(offset+2),:], y:y_data[offset:(offset+2)]}
        # so we use all data during training
        feed_dict = {x: x_data[:,:], y: y_data[:]}
        _, l = session.run([train_step, loss],feed_dict=feed_dict)
        if (i+1) % 100 == 0:
            print("--- %d: Loss = %lf" % (i+1, l))
    # Let's validate if we get the correct output given an input
    print("In: ")
    # You can choose all inputs (0:4) or some by modifying the range eg (1:2)
    input = x_data[0:4,:]
    print(input)
    hidden = tf.matmul(input, W0) + b0
    print("Predicted output:")
    yp = tf.matmul(tf.nn.relu(hidden), W1) + b1
    # yp = tf.matmul(hidden, W1) + b1
    print(print(1*np.greater(yp.eval(),0.25)))
