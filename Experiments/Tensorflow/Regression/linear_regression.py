'''
Linear Regression using Numerical Optimization on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python linear_regression.py
# Prerequisite: tensorflow (see tensorflow.org)
#             : matplotlib (http://matplotlib.org/)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Tunable parameters (Try changing the values and see what happens)
# Variable samples >= 3 ; stddev > 0.; xcoeff are real numbers
samples = 100
stddev = 1.0
# xcoeff should be predicted by solving y = A*x using SVD; try changing the values
xcoeff = tf.transpose(tf.constant([[2., -3.5, 12.5]]))

# The computation
# We get x by sampling a normal dist
x = tf.random_normal([samples,1],stddev=stddev)
# A = tf.Variable(tf.concat(1,[tf.concat(1,[x*x,x]),tf.ones_like(x)]))
# x = tf.constant([ [3.5], [2.5], [1.], [-10.]  ])
# A = tf.constant(tf.concat(1,[tf.concat(1,[x*x,x]),tf.ones_like(x)]))

# xp = tf.Variable(tf.random_uniform([3,1], -10.0, 10.0))
x1 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
x2 = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
x3 = tf.Variable(tf.zeros([1]))

# Output
yp = x1*x*x + x2*x + x3 # tf.matmul(A,xp)
y = 2.*x*x -3.*x + 12.5 # tf.matmul(A,xcoeff)

loss = tf.reduce_mean(tf.square(yp - y))
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(1000):
        session.run(train_step)
        if ((i+1) % 50) == 0:
            print(i+1, session.run(x1), session.run(x2), session.run(x3))
