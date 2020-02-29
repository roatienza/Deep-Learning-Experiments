#!/usr/bin/env python
# coding: utf-8

# ## MLP MNIST with data augmentation

# In[ ]:


'''
MLP network for MNIST digits classification w/ data augment
Test accuracy: 97.7
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# numpy package
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

from numpy.random import seed
seed(12345)
import tensorflow as tf
tf.random.set_seed(12345)

# load mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# compute the number of labels
num_labels = np.amax(y_train) + 1
# convert to one-hot vector
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# image dimensions (assumed square)
image_size = x_train.shape[1]
input_size = image_size * image_size
# we train our network using float data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
batch_size = 128
hidden_units = 256
data_augmentation = True
epochs = 20
max_batches = len(x_train) / batch_size

# this is 3-layer MLP with ReLU after each layer
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_size))
model.add(Activation('relu'))
model.add(Dense(hidden_units))
model.add(Activation('relu'))
model.add(Dense(num_labels))
# this is the output for one-hot vector
model.add(Activation('softmax'))
model.summary()

# loss function for one-hot vector
# use of sgd optimizer
# accuracy is good metric for classification tasks
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

# validate the model on test dataset to determine generalization
# score = model.evaluate(x_test, y_test, batch_size=batch_size)
# print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))

# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    # train the network no data augmentation
    x_train = np.reshape(x_train, [-1, input_size])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    # we need [width, height, channel] dim for data aug
    x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)
    # fits the model on batches with real-time data augmentation:
    #model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
    #                    steps_per_epoch=len(x_train) / 32, 
    #                    epochs=epochs)
    for e in range(epochs):
        print('Epoch', e)
        batches = 0
        for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=batch_size):
            x_batch = np.reshape(x_batch, [-1, image_size*image_size])
            model.fit(x_batch, y_batch)
            batches += 1
            if batches >= len(x_train) / 32:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                break

# Score trained model.
x_test = np.reshape(x_test, [-1, input_size])
scores = model.evaluate(x_test,
                        y_test,
                        batch_size=batch_size,
                        verbose=False)
print('Test loss:', scores[0])
print('Test accuracy: %0.1f%%' % (100 * scores[1]) )


# In[ ]:




