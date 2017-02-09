'''
Deep Neural Networks on Not MNIST
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python notmnist_mlp.py
# Prerequisite: tensorflow (see tensorflow.org)

from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import sys
import tarfile

from os import walk
from os.path import join
from mnist_library import readfile
from six.moves.urllib.request import urlretrieve

# datadir = 'notMNIST_small'

train_dir = 'notMNIST_small'
test_dir = 'notMNIST_small'

train_dirnames = []

# filenames = []
# dirnames = []
# imagedirs = []

# samples = 10

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
    data_folders = [os.path.join(root, d)
                    for d in sorted(os.listdir(root)) if d != '.DS_Store']
    if len(data_folders) == num_classes:
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

train_folders = extract(train_filename)
test_folders = extract(test_filename)

def get_dirs(dir):
    for (dirpath, dirs, files) in walk(dir):
        # files = [f for f in files if not f[0] == '.']
        return [d for d in dirs if not d[0] == '.']
        # dirnames.extend(dirs)

def read_dir_image_files(dirs):
    # dirs = extract(train_filename)
    imagelabels = []
    imagedata = []
    for dir in dirs:
        # dir is label
        # imagedir = join(datadir,dir)
        for (path, folders, files) in walk(dir):
            print("Reading folder ", dir, "...")
            files = [f for f in files if not f[0] == '.']
            label =  ( np.arange(num_classes) == (ord(dir[0])-ord('A')) ).astype(np.float32)
            for f in files:
                data = readfile(join(dir,f))
                if (data.size > 0):
                    imagelabels.append(label)
                    imagedata.append(data)
            break

    return np.array(imagedata),np.array(imagelabels)

# train_labels = np.array(imagelabels)
# train_dataset = np.array(imagedata)

train_dataset, train_labels = read_dir_image_files(train_folders)
test_dataset, test_labels = read_dir_image_files(test_folders)

num_labels =  train_labels.shape[1]
image_size = train_dataset.shape[2]

train_dataset = train_dataset.reshape((-1,image_size*image_size)).astype(np.float32)
test_dataset = test_dataset.reshape((-1,image_size*image_size)).astype(np.float32)

# test_dataset = train_dataset
# test_labels = train_labels

print(train_dataset.shape)
print(train_labels.shape)
print(test_dataset.shape)
print(test_labels.shape)

# Test accuracy: 99.9% at 1024*16 hidden units
batch_size = 128
hidden_units = 1024*16

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
        tf.truncated_normal([hidden_units, num_labels]))
    biases2 = tf.Variable(tf.zeros([num_labels]))

    def model(data):
        logits1 = tf.matmul(data, weights1) + biases1
        relu1 = tf.nn.relu(logits1)
        logits = tf.matmul(relu1, weights2) + biases2
        return logits, tf.nn.softmax(logits)

    train_logits, train_pred = model(tf_train_dataset)
    loss = tf.reduce_mean(
         tf.nn.softmax_cross_entropy_with_logits(train_logits, tf_train_labels))
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    # tf_test_dataset = tf.constant(readfile(testfile).reshape(-1,image_size*image_size).astype(np.float32))
    test_logits, test_pred = model(test_dataset)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) /
            predictions.shape[0])

num_steps = 10001

with tf.Session(graph=graph) as session:
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
    # print(np.rint(session.run(test_pred)))
