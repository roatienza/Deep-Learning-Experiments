'''
Linear Algebra on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/linear_algebra.py
'''
# On command line: python linear_algebra.py
# Prerequisite: tensorflow (see tensorflow.org)

import tensorflow as tf
import sys

# Square matrix A of rank 2
A = tf.constant([ [1.,2.], [3.,4.] ])

# 2x2 Square matrix B will all elements 2.
B = tf.fill([2,2], 2.)

# 2x1 matrix will all elements equal to 1
x = tf.ones([2,1])

# 2x1 matrix will all elements equal to 2.0
b = tf.fill([2,1], 2.)

# matrix multiplication
y1 = tf.matmul(A,x)

# addition of matrix
y2 = tf.add(y1,b)

# run within a session and print
with tf.Session() as session:
	print("Tensorflow version: " + tf.__version__)
	tf.global_variables_initializer().run()
	print("y1 = A * x ")
	sys.stdout.write(str(y1.eval()) + " = " + str(A.eval()) + " * " + str(x.eval()) + "\n")
	print("y2 = A*x + b ")
	sys.stdout.write(str(y2.eval()) + " = " + str(A.eval()) + " * " + str(x.eval()) + " + " + str(b.eval()) + "\n")

	# transpose of A
	print("tran(A) = ")
	print(tf.matrix_transpose(A).eval())

	# inverse of A
	print("inv(A) = ")
	print(tf.matrix_inverse(A).eval())

	# rank of A
	print("rank(A) = ")
	print(tf.rank(A).eval())

	# product of a matrix and its inverse is an identity (non-singular)
	print("A*inv(A) = Eye(2)")
	print( tf.matmul(A,tf.matrix_inverse(A)).eval() )

	# element-wise multiplication
	print("elem(A)*elem(B) = ")
	print(tf.mul(A,B).eval())

	# element-wise addition
	print("elem(A)+elem(B) = ")
	print(tf.add(A,B).eval())