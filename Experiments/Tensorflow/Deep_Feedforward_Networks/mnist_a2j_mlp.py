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

train_dataset = all_dataset
train_labels = all_labels

print("Training size: ", train_dataset.shape)
print("Training labels: ", train_labels.shape)
print("Test size: ", test_dataset.shape)
print("Test labels: ", test_labels.shape)

num_labels = train_labels.shape[1]
num_data = train_labels.shape[0]

image_size = 28
batch_size = 64
hidden_units = 512*4
learning_rate = 0.0002
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
         tf.nn.softmax_cross_entropy_with_logits(logits=train_logits, labels=tf_train_labels))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
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
        # rand_offset = np.random.randint(0,batch_size/2)
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
            print("Test accuracy: %.1f%%" % accuracy(test_pred.eval(),
                                                          test_labels))
    print(np.rint(test_pred.eval()))
