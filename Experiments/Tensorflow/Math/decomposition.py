'''
Decomposition on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/decomposition.py
'''
# On command line: python decomposition.py
# Prerequisite: tensorflow (see tensorflow.org)

import tensorflow as tf
import numpy as np
import numpy.linalg as la

print("Tensorflow version: " + tf.__version__)
# Real symmetric matrix S of rank 2
S = tf.constant([ [1.,2.], [2.,1.] ])
print("S = ")
print(S.eval(session=tf.Session()))

# Eigen Decomposition
e,Q = tf.self_adjoint_eig(S)
# diagonal matrix made of eigenvalues of S
V = tf.diag(e)
# S_ = S since S = Q*V*tran(Q) for real symmetric matrix
S_ = tf.matmul(Q,tf.matmul(V,Q))
print("S_ = S = ")
tf.Print(S,[S])
print(S_.eval(session=tf.Session()))

# Frobenius, Euclidean or L2 norm using np - unfortunately tf does not have a function to compute L1 norm
print("l2(S) =")
print(la.norm(S.eval(session=tf.Session())))

# SVD decomposition
d, U, V1 = tf.svd(S, full_matrices=True, compute_uv=True)
# U and V1 are orthogonal matrices; I must be therefore an identity matrix
I = tf.matmul(U,tf.transpose(V1))
print("I = ")
print(I.eval(session=tf.Session()))
D = tf.diag(d)
# S_ = S since S = U*D*tran(V1)
print("S_ = S = ")
S_ = tf.matmul(U,tf.matmul(D,tf.transpose(V1)))
print(S_.eval(session=tf.Session()))
# Moore-Penrose pseudoinverse
D = tf.diag(np.reciprocal(d))
print("pseudo_inv(S) = ")
S_ = tf.matmul(U,tf.matmul(D,tf.transpose(V1)))
print(S_.eval(session=tf.Session()))

# inverse of S BUT applicable to non-singular square matrices only
print("inv(S) = ")
print(tf.matrix_inverse(S).eval(session=tf.Session()))
