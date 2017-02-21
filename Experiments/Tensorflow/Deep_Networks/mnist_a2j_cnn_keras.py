"""
CNN on Not MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
"""
# On command line: python3 mnist_a2j_cnn_keras.py
# Prerequisite: tensorflow 1.0 and keras
# must run mnist_a2j_2pickle.py first (one-time) to generate the data

from __future__ import print_function

import numpy as np
import pickle
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

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

print("Training size: ", train_dataset.shape)
print("Training labels: ", train_labels.shape)
print("Test size: ", test_dataset.shape)
print("Test labels: ", test_labels.shape)

# small batch size appears to work
batch_size = 128
depth1 = 32
depth2 = 64

# already figured out by keras
# hidden_units1 = int((depth2*image_size/4*image_size/4))
hidden_units2 = 1024
patch_size = 3
dropout = 0.8
learning_rate = 0.001

model = Sequential()
# input: 28x28 images with 1 channel -> (28, 28, 1) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(depth1, patch_size, patch_size, border_mode='same', input_shape=(image_size, image_size, channel)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(depth2, patch_size, patch_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(hidden_units2))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

adam = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(train_dataset, train_labels,
          nb_epoch=10,
          batch_size=batch_size, shuffle=False)
score = np.asarray(model.evaluate(test_dataset, test_labels, batch_size=batch_size))*100.0
# Accuracy: 98.0%
print("\nTest accuracy: %.1f%%" % score[1])
print("Elapsed: " , elapsed(time.time() - start_time))
