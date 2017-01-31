'''
Linear Regression by SVD on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python linear_inv.py
# Prerequisite: tensorflow (see tensorflow.org)
#             : matplotlib (http://matplotlib.org/)

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Tunable parameters (Try changing the values and see what happens)
# Variable samples >= 3 ; stddev > 0.; xcoeff are real numbers
samples = 100
stddev = 1.0
# xcoeff should be predicted by solving y = A*x using SVD; try changing the values
xcoeff = tf.transpose(tf.constant([[2., -3.5, 12.5]]))

# The computation
# We get elements of A by sampling a normal distribution
a = tf.Variable(tf.random_normal([1, samples],stddev=stddev))
# Sort to produce a nice plot later
b, _ = tf.nn.top_k(a,k=samples)
# Correct the shape
a = tf.reshape(b,[samples,1])

# Inputs to form y = a*a*xcoeff[0] + a*xcoeff[1] + xcoeff[2]
A = tf.concat(1,[tf.concat(1,[a*a,a]),tf.ones_like(a)])
# Observable outputs
y = tf.matmul(A,xcoeff)

# SVD - Singular Value Decomposition
d, U, V = tf.svd(A, full_matrices=True, compute_uv=True)

# Wondering why tensorflow does not generate the diagonal matrix directly
# D is the diagonal matrix with the diagonal elements being the reciprocal of d
D = tf.diag(np.reciprocal(d))

# D is actually nxm so zero padding is needed
r = D.get_shape().as_list()[0]
Z = tf.zeros([r,samples-r])
D = tf.concat(1,[D,Z])

# This is linear regression by SVD
# Ax = y  mxn nx1 = mx3 3x1 = mx1
# x = Inv(A)y = nxm mx1 = 3xm mx1  ; x is our predicted xcoeff
# Inv(A) = A_ = VDtran(U) = nxn nxm mxm = nxm = 3x3 3xm 3xm = 3xm

# x (predicted xcoeff) is determined using Moore-Penrose pseudoinverse
A_ = tf.matmul(V,tf.matmul(D,tf.transpose(U)))
x = tf.matmul(A_,y)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    # values are the xcoeff
    print("Actual x =")
    print(xcoeff.eval())

    print("\nPredicted x = ")
    print(x.eval())

    # Let's plot; Make all variables 1D by reshaping
    a = tf.reshape(a,[a.get_shape().as_list()[0]])
    y = tf.reshape(y,[y.get_shape().as_list()[0]])
    # Predicted model, yp, based on x
    yp = tf.matmul(A,x)
    yp = tf.reshape(yp,[yp.get_shape().as_list()[0]])
    plt.plot(a.eval(), y.eval(), 'ro', a.eval(), yp.eval(), 'b')
    red = mpatches.Patch(color='red', label='Data')
    blue = mpatches.Patch(color='blue', label='Model')
    plt.legend(handles=[red,blue])
    plt.show()
