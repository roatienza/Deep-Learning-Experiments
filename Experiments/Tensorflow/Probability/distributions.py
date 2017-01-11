'''
Common Distributions on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python distributions.py
# Prerequisite: tensorflow (see tensorflow.org)
#             : matplotlib (http://matplotlib.org/)

import tensorflow as tf
import matplotlib.pyplot as plt

# Normal and Uniform are common distributions used to generate weights and biases

print("Tensorflow version: " + tf.__version__)
# Create a N X N weights/samples from Gaussian/Normal dist; mean=0.0
A = tf.Variable(tf.truncated_normal([10,10], stddev=2.0))

# Create a N X N X N weights/samples from Uniform dist
B = tf.Variable(tf.random_uniform([10,10,10], minval=0, maxval=2.0 ))

# plot
with tf.Session() as session:
	tf.global_variables_initializer().run()
	s = A.get_shape().as_list()
	A = tf.reshape(A,[s[0]*s[1]])
	#plot shows truncated normal values do not exceed [-2*stddev,2*stddev]
	plt.plot(A.eval())
	plt.ylabel('Values')
	plt.xlabel('n')
	plt.show()

	#plot shows random uniform values range from minval to maxval
	s = B.get_shape().as_list()
	B = tf.reshape(B,[s[0]*s[1]*s[2]])
	plt.plot(B.eval())
	plt.ylabel('Values')
	plt.xlabel('n')
	plt.show()
