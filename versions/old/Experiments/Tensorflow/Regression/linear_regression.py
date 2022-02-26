'''
Linear Regression by Machine Learning on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python3 linear_regression.py
# Prerequisite: tensorflow 1.0 (see tensorflow.org)
#             : matplotlib (http://matplotlib.org/)

from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Tunable parameters (Try changing the values and see what happens)
# Variable samples >= 3 ; stddev > 0.; xcoeff are real numbers
samples = 100
stddev = 1.0
# xcoeff should be predicted by the model, yp
xcoeff = tf.transpose(tf.constant([[2., -3.5, 12.5]]))
learning_rate = 0.1

# The computation
# We get elements of A by sampling a normal distribution
# a = tf.random_normal([samples,1],stddev=stddev)
a = tf.random_normal([1, samples],stddev=stddev)
# a = tf.Variable(tf.random_normal([1, samples],stddev=stddev))
# Sort to produce a nice plot later
b, _ = tf.nn.top_k(a,k=samples)
# Correct the shape
a = tf.reshape(b,[samples,1])

# Inputs to form y = a*a*xp[0] + a*xp[1] + xp[2], xp[] are the weights
A = tf.concat([tf.concat([a*a,a],1),tf.ones_like(a)],1)

# Initial guess on coefficients of predicted linear model
xp = tf.Variable(tf.random_uniform([3,1], -1.0, 1.0))

# Predicted Model
yp = tf.matmul(A,xp)

# Observed outputs
y = tf.matmul(A,xcoeff)

# The smaller the loss, the closer our prediction to the observed outputs
# The loss model used is square of error (yp - y)
# Miinimization of loss is done by Gradient Descent
loss = tf.reduce_mean(tf.square(yp - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(100):
        session.run(train_step)
        if ((i+1) % 10) == 0:
            print("%d : Loss=%0.1lf,  Predicted Parameters = %s" % (i+1, loss.eval(), session.run(xp)))
    # Let's plot
    # Note we have to resample a and save in a constant array a
    # Before this, everytime you call a.eval(), it is resampled
    a = np.array(a.eval())
    A = tf.concat([tf.concat([a*a,a],1),tf.ones_like(a)],1)
    yp = tf.matmul(A,xp)
    y = tf.matmul(A,xcoeff)
    plt.plot(a, y.eval(), 'ro', a, yp.eval(), 'b')
    red = mpatches.Patch(color='red', label='Data')
    blue = mpatches.Patch(color='blue', label='Model')
    plt.legend(handles=[red,blue])
    plt.show()