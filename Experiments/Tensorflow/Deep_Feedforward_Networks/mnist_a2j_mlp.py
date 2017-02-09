"""
Deep Neural Networks on Not MNIST
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
"""
# On command line: python mnist_a2j_mlp.py
# Prerequisite: tensorflow (see tensorflow.org)

from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import sys
import tarfile
import random

from os import walk
from os.path import join
from mnist_library import readfile
from six.moves.urllib.request import urlretrieve

url = 'http://yaroslavvb.com/upload/notMNIST/'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        print('Downloading ', filename, " ...")
        filename, _ = urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified', filename)
        else:
            raise Exception('Failed to verify' +
                            filename + '. Can you get to it with a browser?')
    else:
        print('Found and verified', filename)
    return filename

train_filename = maybe_download('notMNIST_large.tar.gz', 247336696)
test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
num_classes = 10

def extract(filename):
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    data_folders = []
    if os.path.exists(root):
        data_folders = [os.path.join(root, d)
                        for d in sorted(os.listdir(root)) if d != '.DS_Store']
    if len(data_folders) == num_classes:
        print("Using previously extracted files...")
        print(data_folders)
        return data_folders
    tar = tarfile.open(filename)
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    sys.stdout.flush()
    tar.extractall()
    tar.close()
    data_folders = [os.path.join(root, d)
                    for d in sorted(os.listdir(root)) if d != '.DS_Store']
    if len(data_folders) != num_classes:
        raise Exception(
            'Expected %d folders, one per class. Found %d instead.' % (
                num_classes, len(data_folders)))
    print(data_folders)
    return data_folders

def getfiles_fromlist(dirs):
    files = []
    for dir in dirs:
        files.extend([os.path.join(dir,f) for f in sorted(os.listdir(dir)) if f != '.DS_Store'])
    return files

train_folders = extract(train_filename)
test_folders = extract(test_filename)
train_files = np.array(getfiles_fromlist(train_folders))
test_files = np.array(getfiles_fromlist(test_folders))
random.shuffle(train_files)

def read_image_files(files):
    imagelabels = []
    imagedata = []
    for file in files:
        parent_dir = os.path.dirname(file)
        label =  (np.arange(num_classes) == (ord(parent_dir[-1])-ord('A')) ).astype(np.float32)
        data = readfile(file)
        if (data.size > 0):
            imagelabels.append(label)
            imagedata.append(data)
    return np.array(imagedata),np.array(imagelabels)

train_dataset, train_labels = read_image_files(train_files)
test_dataset, test_labels = read_image_files(test_files)

num_labels =  train_labels.shape[1]
image_size = train_dataset.shape[2]

train_dataset = train_dataset.reshape((-1,image_size*image_size)).astype(np.float32)
test_dataset = test_dataset.reshape((-1,image_size*image_size)).astype(np.float32)

print(train_dataset.shape)
print(train_labels.shape)
print(test_dataset.shape)
print(test_labels.shape)

batch_size = 512*4
hidden_units = 512*24

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size *
                                             image_size))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

    weights1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, hidden_units]))
    biases1 = tf.Variable(tf.zeros([hidden_units]))

    weights2 = tf.Variable(
        tf.truncated_normal([hidden_units, hidden_units]))
    biases2 = tf.Variable(tf.zeros([hidden_units]))

    weights3 = tf.Variable(
        tf.truncated_normal([hidden_units, num_labels]))
    biases3 = tf.Variable(tf.zeros([num_labels]))

    def model(data):
        logits1 = tf.matmul(data, weights1) + biases1
        relu1 = tf.nn.relu(logits1)
        logits2 = tf.matmul(relu1, weights2) + biases2
        relu2 = tf.nn.relu(logits2)
        logits = tf.matmul(relu2, weights3) + biases3
        return logits, tf.nn.softmax(logits)

    train_logits, train_pred = model(tf_train_dataset)
    loss = tf.reduce_mean(
         tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

    test_logits, test_pred = model(test_dataset)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])

num_steps = 10001

with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=True)) as session:
    tf.global_variables_initializer().run()
    print(tf.__version__)
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset:
                     batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run(
            [optimizer, loss, train_pred], feed_dict=feed_dict)
        if (step % 500 == 0):
            print("Minibatch loss at step %d: %f" % (step, l))
            print("Minibatch accuracy: %.1f%%" % accuracy(predictions,
                                                          batch_labels))
            print("Test accuracy: %.1f%%" % accuracy(test_pred.eval(),
                                                          test_labels))

    print(np.rint(test_pred.eval()))



# Minibatch accuracy: 92.6%
# Test accuracy: 93.5%
# batch_size = 256
# hidden_units = 512*16