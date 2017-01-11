'''
Linear Regression using Pseudo Inverse on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python linear_inv.py
# Prerequisite: tensorflow (see tensorflow.org)
#             : matplotlib (http://matplotlib.org/)

import tensorflow as tf
import numpy as np

# variable samples >= 3 ; stddev > 0
samples = 300
stddev = 1.0
x1 = tf.Variable(tf.random_normal([samples,1],stddev=stddev))
y = 2.0*x1*x1 - 3.0*x1 - 12.0

x2 = tf.mul(x1,x1)
x0 = tf.ones_like(x1)
A = tf.concat(1,[tf.concat(1,[x2,x1]),x0])
d, U, V = tf.svd(A, full_matrices=True, compute_uv=True)
D = tf.diag(np.reciprocal(d))
r = D.get_shape().as_list()[0]
Z = tf.zeros([r,samples-r])
D = tf.concat(1,[D,Z])
# the coefficients of y are determined using Moore-Penrose pseudoinverse
A_ = tf.matmul(V,tf.matmul(D,tf.transpose(U)))
x = tf.matmul(A_,y)

# Ax = y  mx3 3x1 = mx1
# x = Inv(A)y = 3xm mx1
# Inv(A) = VDtran(U) = nxn nxm mxm = nxm = 3x3 3xm 3xm = 3xm

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    # values are the coefficents of y
    print(x.eval())

