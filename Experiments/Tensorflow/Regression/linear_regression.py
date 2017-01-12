'''
Linear Regression by Training Parameters on TensorFlow
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
# xcoeff should be predicted by minimizing loss during training
xcoeff = tf.transpose(tf.constant([[2., -3.5, -12.5]]))
learning_rate = 0.1

# The computation
# We get x by sampling a normal dist
x = tf.random_normal([samples,1],stddev=stddev)
A = tf.concat(1,[tf.concat(1,[x*x,x]),tf.ones_like(x)])

xp = tf.Variable(tf.random_uniform([3,1], -1.0, 1.0))

# Output
yp = tf.matmul(A,xp)
y = tf.matmul(A,xcoeff)

loss = tf.reduce_mean(tf.square(yp - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(1000):
        session.run(train_step)
        if ((i+1) % 10) == 0:
            print("%d : Loss=%0.1lf,  Predicted Parameters = %s" % (i+1, loss.eval(), session.run(xp)))
    # Let's plot
    # Note we have to resample x and save in a constant array x
    # Before this, everytime you call x.eval(), it is resampled
    x = np.array(x.eval())
    A = tf.concat(1,[tf.concat(1,[x*x,x]),tf.ones_like(x)])
    yp = tf.matmul(A,xp)
    y = tf.matmul(A,xcoeff)
    plt.plot(x, y.eval(), 'r.', x, yp.eval(), 'bx')
    plt.show()