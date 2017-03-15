'''
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.

Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time

start_time = time.time()
def elapsed(sec):
    if sec<60:
        return str(sec) + " sec"
    elif sec<(60*60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


# Target log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

training_file = 'belling_the_cat.txt'

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content,[content.shape[1],])
    return content

training_data = read_data(training_file)
print("Loaded training data...")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reverse_dictionary

count, word_dict, word_rev_dict = build_dataset(training_data)
vocab_size = len(word_dict)

# Parameters
learning_rate = 0.001
training_iters = 150000
display_step = 1000
n_input = 3

# number of units in RNN cell
n_hidden = 512

# tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, weights, biases):
    x = tf.reshape(x, [-1, n_input])
    x = tf.split(x,n_input,1)

    # Define a lstm cell with tensorflow
    # rnn_cell = rnn.BasicLSTMCell(n_hidden)
    # rnn_cell = rnn.BasicRNNCell(n_hidden)
    rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden),rnn.BasicLSTMCell(n_hidden)])
    # rnn_cell = rnn.DropoutWrapper(rnn_cell,input_keep_prob=0.8,output_keep_prob=1.0)
    # Get rnn cell output
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model#
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as session:
    session.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

    writer.add_graph(session.graph)

    while step < training_iters:
        # Generate a minibatch.
        if offset > (len(training_data)-end_offset):
            offset = random.randint(0, n_input+1)

        words_in_keys = [ [word_dict[ str(training_data[i])]] for i in range(offset, offset+n_input) ]
        words_in_keys = np.reshape(np.array(words_in_keys), [-1, n_input, 1])

        word_out_onehot = np.zeros([vocab_size], dtype=float)
        word_out_onehot[word_dict[str(training_data[offset+n_input])]] = 1.0
        word_out_onehot = np.reshape(word_out_onehot,[1,-1])

        _, acc, loss, onehot_pred = session.run([optimizer, accuracy, cost, pred], \
                                                feed_dict={x: words_in_keys, y: word_out_onehot})
        loss_total += loss
        acc_total += acc
        if (step+1) % display_step == 0:
            print("Iter= " + str(step+1) + ", Average Loss= " + \
                  "{:.6f}".format(loss_total/display_step) + ", Average Accuracy= " + \
                  "{:.5f}".format(acc_total/display_step))
            acc_total = 0
            loss_total = 0
            words_in = [training_data[i] for i in range(offset, offset + n_input)]
            word_out = training_data[offset + n_input]
            word_out_pred = word_rev_dict[int(tf.argmax(onehot_pred, 1).eval())]
            print("%s - [%s] vs [%s]" % (words_in,word_out,word_out_pred))
        step += 1
        offset += (n_input+1)
    print("Optimization Finished!")
    print("Elapsed time: ", elapsed(time.time() - start_time))
    print("Run on command line.")
    print("\ttensorboard --logdir=%s" % (logs_path))
    print("Point your web browser to: http://localhost:6006/")
    while True:
        prompt = "%s words: " % n_input
        sentence = input(prompt)
        sentence = sentence.strip()
        words = sentence.split(' ')
        if len(words) != n_input:
            continue
        words_in_keys = [ word_dict[str(words[j])] for j in range(len(words)) ]
        words_in_keys = np.reshape(np.array(words_in_keys), [-1, n_input, 1])
        onehot_pred = session.run(pred, feed_dict={x: words_in_keys})
        print("Predicted: %s" % word_rev_dict[int(tf.argmax(onehot_pred, 1).eval())])
