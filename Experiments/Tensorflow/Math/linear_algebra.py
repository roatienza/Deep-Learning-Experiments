'''
Linear Algebra on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/linear_algebra.py
'''
# On command line: python linear_algebra.py
# Prerequisite: tensorflow (see tensorflow.org)

import tensorflow as tf
import numpy as np
import sys

A = tf.constant([ [1.,2.], [3.,4.] ])
B = tf.fill([2,2], 2.)
x = tf.ones([2,1])
b = tf.fill([2,1], 2.)
y1 = tf.matmul(A,x)
y2 = tf.add(y1,b)

# run within a session and print
with tf.Session() as session:
	print("Tensorflow version: " + tf.__version__)
	tf.global_variables_initializer().run()
	print("y1 = A * x ")
	sys.stdout.write(str(y1.eval()) + " = " + str(A.eval()) + " * " + str(x.eval()) + "\n")
	print("y2 = A*x + b ")
	sys.stdout.write(str(y2.eval()) + " = " + str(A.eval()) + " * " + str(x.eval()) + " + " + str(b.eval()) + "\n")
	print("tran(A) = ")
	print(tf.matrix_transpose(A).eval())
	print("inv(A) = ")
	print(tf.matrix_inverse(A).eval())
	print("rank(A) = ")
	print(tf.rank(A).eval())
	print("A*inv(A) = Eye(2)")
	print( tf.matmul(A,tf.matrix_inverse(A)).eval() )
	print("elem(A)*elem(B) = ")
	print(tf.mul(A,B).eval())
	print("elem(A)+elem(B) = ")
	print(tf.add(A,B).eval())