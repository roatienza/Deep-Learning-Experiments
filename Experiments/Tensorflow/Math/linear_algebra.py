'''
Linear Algebra on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/linear_algebra.py
'''
# On command line: python linear_algebra.py
# Prerequisite: tensorflow (see tensorflow.org)

import tensorflow as tf
import numpy as np

A = tf.Variable(np.array([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ]))
B = tf.ones([3,3])
b = tf.fill([1,3], 2)

# run within a session and print
with tf.Session() as session:
	print("Tensorflow version: " + tf.__version__)
	tf.global_variables_initializer().run()
	print("%s", A.eval())
	print(B.eval())
	print(b.eval())

