'''
Hello World on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python hello.py
# Prerequisite: tensorflow (see tensorflow.org)

import tensorflow as tf

# create a tensorflow constant string
hello = tf.constant('Hello World!')

# run within a session and print
with tf.Session() as session:
    print("Tensorflow version: " + tf.__version__)
    tf.global_variables_initializer().run()
    print(hello.eval())
