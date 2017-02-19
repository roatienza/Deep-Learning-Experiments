'''
Principal Component Analysis
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python3 pca.py
# Prerequisite: tensorflow 1.0 (see tensorflow.org)


from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math

print("Tensorflow version: " + tf.__version__)

# let's represent digits 1..5 by black(1)/white(0) 5x5pix images
# Try tweaking the image after the first run of observation

one_1 = np.array([  [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 0., 0., 0.]
                ])

one_2 = np.array([  [0., 0., 0., 1., 1.],
                    [0., 0., 0., 1., 1.],
                    [0., 0., 0., 1., 1.],
                    [0., 0., 0., 1., 1.],
                    [0., 0., 0., 1., 1.]
                ])

two_1 = np.array([  [1., 1., 1., 1., 1.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [1., 1., 1., 1., 1.]
                ])
two_2 = np.array([  [0., 1., 1., 1., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 1., 1., 0.]
                ])

three_1 = np.array([[1., 1., 1., 1., 1.],
                    [0., 0., 1., 1., 1.],
                    [1., 1., 1., 1., 1.],
                    [0., 0., 1., 1., 1.],
                    [1., 1., 1., 1., 1.]
                ])
three_2 = np.array([[1., 1., 1., 1., 0.],
                    [0., 0., 0., 1., 0.],
                    [1., 1., 1., 1., 0.],
                    [0., 0., 0., 1., 0.],
                    [1., 1., 1., 1., 0.]
                ])

four_1 = np.array([ [1., 0., 0., 0., 1.],
                    [1., 0., 0., 0., 1.],
                    [1., 1., 1., 1., 1.],
                    [0., 0., 0., 0., 1.],
                    [0., 0., 0., 0., 1.]
                ])
four_2 = np.array([ [1., 1., 0., 0., 1.],
                    [1., 1., 0., 0., 1.],
                    [1., 1., 1., 1., 1.],
                    [0., 0., 0., 0., 1.],
                    [0., 0., 0., 0., 1.]
                ])

five_1 = np.array([ [1., 1., 1., 1., 0.],
                    [1., 0., 0., 0., 0.],
                    [1., 1., 1., 1., 0.],
                    [0., 0., 0., 1., 0.],
                    [1., 1., 1., 1., 0.]
                ])
five_2 = np.array([ [0., 1., 1., 1., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 1., 1., 1., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 1., 1., 1., 0.]
                ])


# Let's observe the largest eigenvector of each image

w, v = np.linalg.eig(one_1)
max = np.argmax(w)
print("1 eig: ", v[:,max])
w, v = np.linalg.eig(one_2)
max = np.argmax(w)
print("1 eig: ", v[:,max])
print("")

w, v = np.linalg.eig(two_1)
max = np.argmax(w)
print("2 eig: ", v[:,max])
w, v = np.linalg.eig(two_2)
max = np.argmax(w)
print("2 eig: ", v[:,max])
print("")

w, v = np.linalg.eig(three_1)
max = np.argmax(w)
print("3 eig: ", v[:,max])
w, v = np.linalg.eig(three_2)
max = np.argmax(w)
print("3 eig: ", v[:,max])
print("")

w, v = np.linalg.eig(four_1)
max = np.argmax(w)
print("4 eig: ", v[:,max])
w, v = np.linalg.eig(four_2)
max = np.argmax(w)
print("4 eig: ", v[:,max])
print("")

w, v = np.linalg.eig(five_1)
max = np.argmax(w)
print("5 eig: ", v[:,max])
w, v = np.linalg.eig(five_2)
max = np.argmax(w)
print("5 eig: ", v[:,max])
print("")

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
