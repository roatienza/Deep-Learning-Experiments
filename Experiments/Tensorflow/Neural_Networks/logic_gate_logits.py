'''
Logical Operation by 2-layer Neural Networks (Logistic Regression) on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python logic_gate_logits.py
# Prerequisite: tensorflow (see tensorflow.org)


import tensorflow as tf
import numpy as np

learning_rate = 0.4
x_data = np.reshape(np.array( [[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=np.float32 ),[4,2])
# try other logics; xor = [0., 1., 1., 0.], or = [0., 1., 1., 1.], and = [0., 0., 0., 1.], etc
logic_out = np.array([0., 1., 1., 0.], dtype=np.float32)
y_data = np.reshape(logic_out,[4,1])
n = y_data.shape[0]

x = tf.placeholder(tf.float32, shape=(None, 2))
y = tf.placeholder(tf.float32, shape=(None, 1))

# try other values for nhidden
nhidden = 16
W0 = tf.Variable(tf.random_normal([2, nhidden],stddev=0.1))
b0 = tf.Variable(tf.zeros([nhidden]))

W1 = tf.Variable(tf.random_normal([nhidden, 1],stddev=0.1))
b1 = tf.Variable(tf.zeros([1, 1]))

hidden = tf.matmul(x, W0) + b0
yp = tf.matmul(tf.nn.relu(hidden), W1) + b1
logits = tf.nn.softmax(yp,dim=0)

entropy = -tf.mul(y,tf.log(logits))
loss = tf.reduce_mean(entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for i in range(1000):
        # mini-batch can also be used but we a small set of data only
        # offset = (i*2)%(n-2)
        # feed_dict ={x:x_data[offset:(offset+2),:], y:y_data[offset:(offset+2)]}
        # so we use all data during training
        feed_dict = {x: x_data[:,:], y: y_data[:]}
        _, l, y_, yp_ = session.run([train_step, loss, y, logits],feed_dict=feed_dict)
        if (i+1) % 100 == 0:
            print("--- %d: Loss = %lf" % (i+1, l))
    # Let's validate if we get the correct output given an input
    print("In: ")
    # You can choose all inputs (0:4) or some by modifying the range eg (1:2)
    input = x_data[0:4,:]
    print(input)
    hidden = tf.matmul(input, W0) + b0
    print("Predicted output:")
    yp = tf.nn.softmax(tf.matmul(tf.nn.relu(hidden), W1) + b1,dim=0)
    print(print(1*np.greater(yp.eval(),0.25)))
