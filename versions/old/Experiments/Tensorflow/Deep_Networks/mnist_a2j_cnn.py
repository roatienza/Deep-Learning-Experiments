"""
CNN on Not MNIST
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
"""
# On command line: python3 mnist_a2j_cnn.py
# Prerequisite: tensorflow 1.0 (see tensorflow.org)
# must run mnist_a2j_2pickle.py first (one-time) to generate the data

from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle
import time

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"

# use of pickle to speed up loading of data
pickle_file = open( "mnist_a2j.pickle", "rb" )
data = pickle.load(pickle_file)
test_labels = data["test_labels"]
train_labels = data["all_labels"]
test_dataset = data["test_dataset"]
train_dataset = data["all_dataset"]
del data
pickle_file.close()

num_labels = train_labels.shape[1]
num_data = train_labels.shape[0]

image_size = 28
channel = 1
train_dataset = train_dataset.reshape((-1,image_size, image_size, channel)).astype(np.float32)
test_dataset = test_dataset.reshape((-1,image_size, image_size, channel)).astype(np.float32)

print(train_dataset.shape)
print(train_labels.shape)
print(test_dataset.shape)
print(test_labels.shape)

# small batch size appears to work
batch_size = 128
depth1 = 32
depth2 = 64

hidden_units1 = int((depth2*image_size/4*image_size/4))
hidden_units2 = 1024
patch_size = 3
num_steps = 50001
dropout = 0.8
learning_rate = 0.001

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,
                                     shape=(batch_size, image_size , image_size, channel))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    weights0 = tf.Variable(
        tf.truncated_normal([patch_size, patch_size, channel, depth1], stddev=1.0))
    biases0 = tf.Variable(tf.zeros([depth1]))

    weights1 = tf.Variable(
        tf.truncated_normal([patch_size, patch_size, depth1, depth2], stddev=1.0))
    biases1 = tf.Variable(tf.zeros([depth2]))

    weights2 = tf.Variable(
        tf.truncated_normal([hidden_units1, hidden_units2], stddev=0.1))
    biases2 = tf.Variable(tf.constant(1.0, shape=[hidden_units2]))

    weights3 = tf.Variable(
        tf.truncated_normal([hidden_units2, num_labels], stddev=0.1))
    biases3 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    keep_prob = tf.placeholder(tf.float32)

    def model(data, dropout):
        conv1 = tf.nn.conv2d(data, weights0, [1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.bias_add(conv1, biases0)
        pool1 = tf.nn.max_pool(tf.nn.relu(relu1), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        conv2 = tf.nn.conv2d(pool1, weights1, [1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.bias_add(conv2, biases1)
        pool2 = tf.nn.max_pool(tf.nn.relu(relu2), [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])

        logits1 = tf.nn.relu( tf.add(tf.matmul(reshape, weights2), biases2) )
        logits1 = tf.nn.dropout(logits1, dropout)
        logits2 = tf.add( tf.matmul(logits1, weights3), biases3 )

        return logits2, tf.nn.softmax(logits2)

    logits, train_pred = model(tf_train_dataset,dropout=dropout)
    loss = tf.reduce_mean(
         tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    test_logits, test_pred = model(test_dataset,1.0)

def accuracy(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.eval()*100.0

# with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=True)) as session:
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print(tf.__version__)
    for step in range(num_steps):
        offset = (step * batch_size)% (num_data - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset:
                     batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_pred], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch (size=%d) loss at step %d: %f" % (batch_size, step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions,
                                                          batch_labels))
    # Accuracy: 97.4%
    print("Test accuracy: %.1f%%" % accuracy(test_pred.eval(),
                                             test_labels))
    print("Elapsed: ", elapsed(time.time() - start_time))
    # print(np.rint(test_pred.eval()))

