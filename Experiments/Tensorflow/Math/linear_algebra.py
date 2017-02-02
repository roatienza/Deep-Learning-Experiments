'''
Linear Algebra on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python linear_algebra.py
# Prerequisite: tensorflow (see tensorflow.org)

from __future__ import print_function

import tensorflow as tf
import numpy as np

# Square matrix A of rank 2
A = tf.constant([ [1.,2.], [3.,4.] ])

# 2x2 Square, Diagonal, Symmetric matrix B
B = tf.diag([5.,6.])

# 2x2 Square matrix
C = tf.constant([ [1.,2.], [2.,4.] ])

# 2x1 vector will all elements equal to 1
x = tf.ones([2,1])

# 2x1 vector will all elements equal to 2.0
b = tf.fill([2,1], 2.)

# 2x1 vector
y = tf.constant([ [-1.], [1.] ])

# run within a session and print
with tf.Session() as session:
    print("Tensorflow version: " + tf.__version__)
    tf.global_variables_initializer().run()

    print("A = ")
    print(A.eval())

    print("B = ")
    print(B.eval())

    print("C = ")
    print(C.eval())

    print("x = ")
    print(x.eval())

    print("b = ")
    print(b.eval())

    print("y = ")
    print(y.eval())

    # Tensor multiplication
    print("Ax = ")
    print(tf.matmul(A, x).eval())

    # Tensor addition
    print("A + B =")
    print(tf.add(A, B).eval())

    print("A + b =")
    print(tf.add(A, b).eval())

    # Rank of A and B; Number of indices to identify each element
    print("tensorRank(A) = ")
    print(tf.rank(A).eval())
    print("tensorRank(C) = ")
    print(tf.rank(C).eval())

    # Matrix rank
    print("rank(A) = ")
    print(np.linalg.matrix_rank(A.eval()))
    print("rank(C) = ")
    print(np.linalg.matrix_rank(C.eval()))

    # Transpose
    print("tran(A) = ")
    print(tf.matrix_transpose(A).eval())
    print("tran(B) = ")
    print(tf.matrix_transpose(B).eval())

    # Inverse
    print("inv(A) = ")
    print(tf.matrix_inverse(A).eval())
    # Inverse of diagonal matrix has diag elements of the reciprocal of diag elements B
    print("inv(B) = ")
    print(tf.matrix_inverse(B).eval())
    print("inv(C) = ") # since C has rank 1, this will cause error
    try:
        print(tf.matrix_inverse(C).eval())
    except:
        print("C is not invertible")

    # Product of a matrix and its inverse is an identity (non-singular)
    print("A*inv(A) = Eye(2)")
    print( tf.matmul(A,tf.matrix_inverse(A)).eval() )

    # Element-wise multiplication
    print("elem(A)*elem(B) = ")
    print(tf.mul(A,B).eval())

    # Element-wise addition
    print("elem(A)+elem(B) = ")
    print(tf.add(A,B).eval())

    # Dot product
    print("x dot b")
    print(tf.matmul(x,b,transpose_a=True).eval())

    # Identity matrix of same shape as A
    print("eye(A) = ")
    I = tf.eye(A.get_shape().as_list()[0],A.get_shape().as_list()[1])
    print(I.eval())

    # Multiply eye(A) and A = A
    print("eye(A)*A = A = ")
    print(tf.matmul(I,A).eval())
    print("A * eye(A) = A = ")
    print(tf.matmul(A, I).eval())

    # l1, l2, Frobenius norm
    print("l1(x) = ")
    print(tf.reduce_sum(tf.abs(x)).eval())
    print("l2(x) = ")
    print(tf.sqrt(tf.reduce_sum(tf.square(x))).eval())
    print("Frobenius(A) = ")
    print(tf.sqrt(tf.reduce_sum(tf.square(A))).eval())
    print("Numpy l2(x) =")
    print(np.linalg.norm(x.eval(session=tf.Session())))
    print("Numpy Forbenius(A) =")
    print(np.linalg.norm(A.eval(session=tf.Session())))

    # Can you write the L(inf) ?

    # Orthogonal vectors; How do you make x and y orthonormal?
    print("x dot y")
    print(tf.matmul(x,y,transpose_a=True).eval())

    # Eigenvalues and eigenvectors
    print("Numpy Eigenvalues of (A)=")
    e, v = np.linalg.eig(A.eval())
    print(e)
    print("Numpy Eigenvectors of (A)=")
    print(v)

    # Frobenius norm is equal to the trace of A*tran(A)
    print("Frobenius(A) = Tr(A*tran(A) = ")
    print(tf.sqrt(tf.trace(tf.matmul(A,tf.transpose(A)))).eval())

    # Determinant of A is the product of its eigenvalues
    print("det(A)=")
    print(tf.matrix_determinant(A).eval())
    # Determinant from eigenvalues
    print("det(A) as product of eigenvalues")
    print(tf.reduce_prod(e).eval())