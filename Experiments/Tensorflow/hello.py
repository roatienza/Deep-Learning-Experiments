# run as python hello.py
# Prerequisite : tensorflow (see tensorflow.org)

import tensorflow as tf

# create a tensorflow constant string
hello = tf.constant('Hello World!')

# run within a session and print
with tf.Session() as session:
    print("Tensorflow version: " + tf.__version__)
    tf.global_variables_initializer().run()
    print(hello.eval())
