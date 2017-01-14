'''
Logic Gates by 2-layer Neural Networks on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python logic_gate.py
# Prerequisite: tensorflow (see tensorflow.org)


import tensorflow as tf
import numpy as np

learning_rate = 0.1
x_data = tf.constant(np.reshape([[0., 0.], [0., 1.], [1., 0.], [1., 1.]],[4,2]), dtype=tf.float32)

# xor = [0., 1., 1., 0.], or = [0., 1., 1., 1.], and = [0., 0., 0., 1.], etc
logic_out = np.array([0., 1., 1., 0.])
y_data = tf.constant(np.reshape(logic_out,[4,1]), dtype=tf.float32)

W0 = tf.Variable(tf.random_normal([2, 2], stddev=-1.0))
b0 = tf.Variable(tf.zeros([4, 2]))

W1 = tf.Variable(tf.random_normal([2, 1], stddev=-1.0))
b1 = tf.Variable(tf.zeros([4, 1]))

hidden = tf.matmul(x_data, W0) + b0
yp = tf.matmul(hidden, W1) + b1
loss = tf.reduce_mean(tf.square(yp - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(100):
        _, l = session.run([train_step, loss])
        if (i+1) % 10 == 0:
            print("--- %d: Loss = %lf" % (i, l))
    print("In:")
    print(x_data.eval())
    print("Out:")
    print(tf.round(y_data).eval())
    print("Predicted Out:")
    print(tf.round(yp).eval())