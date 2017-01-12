'''
Linear Regression using Pseudo Inverse on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python linear_inv.py
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
# We get x1 by sampling a normal dist
x1 = tf.Variable(tf.random_normal([samples,1],stddev=stddev))

# Input
A = tf.concat(1,[tf.concat(1,[x1*x1,x1]),tf.ones_like(x1)])
# Output
y = tf.matmul(A,xcoeff)

# SVD
d, U, V = tf.svd(A, full_matrices=True, compute_uv=True)

# Wondering why tensorflow does not generate the diagonal matrix directly
# D is the diagonal matrix with the diagonal elements being the reciprocal of d
D = tf.diag(np.reciprocal(d))

# D is actually nxm so zero padding is needed
r = D.get_shape().as_list()[0]
Z = tf.zeros([r,samples-r])
D = tf.concat(1,[D,Z])

# x (predicted xcoeff) is determined using Moore-Penrose pseudoinverse
A_ = tf.matmul(V,tf.matmul(D,tf.transpose(U)))
x = tf.matmul(A_,y)

# This is linear regression by SVD
# Ax = y  mxn nx1 = mx3 3x1 = mx1
# x = Inv(A)y = nxm mx1 = 3xm mx1  ; x is our predicted xcoeff
# Inv(A) = VDtran(U) = nxn nxm mxm = nxm = 3x3 3xm 3xm = 3xm

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    # values are the xcoeff
    print("Actual x =")
    print(xcoeff.eval())

    x1 = tf.reshape(x1,[x1.get_shape().as_list()[0]])
    y = tf.reshape(y,[y.get_shape().as_list()[0]])

    print("\nPredicted x = ")
    print(x.eval())

    # Let's plot
    yp = tf.matmul(A,x)
    yp = tf.reshape(yp,[yp.get_shape().as_list()[0]])
    plt.plot(x1.eval(), y.eval(), 'r.', x1.eval(), yp.eval(), 'bx')
    plt.show()
