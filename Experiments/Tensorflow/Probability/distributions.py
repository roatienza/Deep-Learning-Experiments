'''
Common Distributions in TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python distributions.py
# Prerequisite: tensorflow (see tensorflow.org)

import tensorflow as tf

# Temporary directory for storing data for plotting using TensorBoard
logs_path = '/tmp/tensorflow_logs/normal'

print("Tensorflow version: " + tf.__version__)

# Generate N=100k samples from Gaussian or Normal dist; mean=0.0, std=1.0
with tf.name_scope('normal'):
    normal_dist = tf.Variable(tf.random_normal([100000]))
    # normal_dist = tf.mul(A,[1])

# Generate N=100k samples from Uniform dist; min=0, max=None
with tf.name_scope('uniform'):
    uniform_dist = tf.Variable(tf.random_uniform([100000]))
    # uniform_dist = tf.mul(B,[1])

# Generate a multinomial with 4 categories (ie 0,1,2,3), 100 samples
with tf.name_scope('multinomial'):
    multi_dist = tf.Variable(tf.multinomial([[1.,1.,1.,1.]],100000))
    # multi_dist = C # tf.mul(C,[1])

# Create a summary to monitor normal dist
tf.histogram_summary("normal", normal_dist)
# Create a summary to monitor uniform dist
tf.histogram_summary("uniform", uniform_dist)
# Create a summary to monitor multinomial dist
tf.histogram_summary("multinomial", multi_dist)

# Merge all summaries into a single op
merged = tf.summary.merge_all()

# Summary writer
with tf.Session() as session:
    tf.global_variables_initializer().run()
    # Logs to Tensorboard
    writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
    for i in range(2):
        _, _, _, summary = session.run([normal_dist,uniform_dist,multi_dist, merged])
        writer.add_summary(summary,i)
    print("Run on command line.")
    print("\ttensorboard --logdir=/tmp/tensorflow_logs/normal ")
    print("Point your web browser to: http://localhost:6006/")