"""
Deep Neural Networks on Not MNIST
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
"""
# On command line: python3 mnist_a2j_mlp.py
# Prerequisite: tensorflow (see tensorflow.org)
# must run mnist_a2j_2pickle.py first (one-time) to generate the data

from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle

# use of pickle to speed up loading of data
pickle_file = open( "mnist_a2j.pickle", "rb" )
data = pickle.load(pickle_file)
test_labels = data["test_labels"]
all_labels = data["all_labels"]
test_dataset = data["test_dataset"]
all_dataset = data["all_dataset"]
del data
pickle_file.close()

# 90% training data, 10% validation data
split = int(0.9*all_dataset.shape[0])
train_dataset = all_dataset[:split,:]
valid_dataset = all_dataset[split+1:,:]
train_labels = all_labels[:split,:]
valid_labels = all_labels[split+1:,:]

print("Training size: ", train_dataset.shape)
print("Training labels: ", train_labels.shape)
print("Validation size: ", valid_dataset.shape)
print("Validation labels: ", valid_labels.shape)
print("Test size: ", test_dataset.shape)
print("Test labels: ", test_labels.shape)

num_labels = train_labels.shape[1]
num_data = train_labels.shape[0]

image_size = 28
batch_size = 32
# batch_sizes = np.array([32, 32, 32, 32, 32, 32, 256])
# change_size_at = 10000
hidden_units = 512*4
learning_rate = 0.0002
# decay_base = 0.8
num_steps = 20001
dropout = 0.8


graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(None, image_size *
                                             image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))

    weights1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, hidden_units]))
    biases1 = tf.add(tf.Variable(tf.zeros([hidden_units])),0.1)

    weights2 = tf.Variable(
        tf.truncated_normal([hidden_units, hidden_units]))
    biases2 = tf.add(tf.Variable(tf.zeros([hidden_units])),0.1)

    weights3 = tf.Variable(
        tf.truncated_normal([hidden_units, num_labels]))
    biases3 = tf.add(tf.Variable(tf.zeros([num_labels])),0.1)

# 3-layer with dropout at hidden layer
    def model(data,dropout=0.5):
        logits1 = tf.add(tf.matmul(data, weights1), biases1)
        relu1 = tf.nn.relu(logits1)
        dropout1 = tf.nn.dropout(relu1, dropout)
        logits2 = tf.add(tf.matmul(dropout1, weights2), biases2)
        relu2 = tf.nn.relu(logits2)
        dropout2 = tf.nn.dropout(relu2, dropout)
        logits = tf.add(tf.matmul(dropout2, weights3), biases3)
        return logits, tf.nn.softmax(logits)

    train_logits, train_pred = model(tf_train_dataset,dropout=dropout)
    loss = tf.reduce_mean(
         tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))

    # Various optimization techniques; SGD has the best performance for this model with the current settings
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # global_step = tf.Variable(0, trainable=False)
    # decaying_learning_rate = tf.train.exponential_decay(learning_rate, global_step,
    #                                            change_size_at, decay_base, staircase=True)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=decaying_learning_rate).minimize(loss, global_step=global_step)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(loss)
    # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(loss)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.2).minimize(loss)

    valid_logits, valid_pred = model(valid_dataset, dropout=1.0)
    test_logits, test_pred = model(test_dataset,dropout=1.0)

def accuracy(predictions, labels):
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(predictions, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy.eval()*100

# with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=True)) as session:
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print(tf.__version__)
    for step in range(num_steps):
        # batch_size = batch_sizes[min(batch_sizes.size-1,(int)(step/change_size_at))]
        rand_offset = np.random.randint(0,batch_size/2)
        offset = ((step * batch_size ) + rand_offset)% (num_data - batch_size)
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
            print("Validation accuracy: %.1f%%" % accuracy(valid_pred.eval(),
                                                          valid_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_pred.eval(),
                                                          test_labels))
    print(np.rint(test_pred.eval()))
