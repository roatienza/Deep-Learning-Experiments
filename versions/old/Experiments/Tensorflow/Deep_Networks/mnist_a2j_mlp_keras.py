"""
Deep Neural Networks on Not MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
"""
# On command line: python3 mnist_a2j_mlp_keras.py
# Prerequisite: tensorflow 1.0 and keras 2.0
# must run mnist_a2j_2pickle.py first (one-time) to generate the data

from __future__ import print_function

import numpy as np
import pickle
import time

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

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

print("Training size: ", train_dataset.shape)
print("Training labels: ", train_labels.shape)
print("Test size: ", test_dataset.shape)
print("Test labels: ", test_labels.shape)

num_labels = train_labels.shape[1]

image_size = 28
input_size = image_size*image_size
batch_size = 128
hidden_units = 512*4
learning_rate = 0.0002
dropout = 0.8

model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

sgd = SGD(lr=learning_rate) # , decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(train_dataset, train_labels,
          epochs=5,
          batch_size=batch_size, shuffle=False)
score = np.asarray(model.evaluate(test_dataset, test_labels, batch_size=batch_size))*100.0
# Accuracy: 86.0%
print("\nTest accuracy: %.1f%%" % score[1])
print("Elapsed: " , elapsed(time.time() - start_time))
