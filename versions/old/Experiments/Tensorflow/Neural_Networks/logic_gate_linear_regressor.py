'''
Logical Operation by 2-layer Neural Networks (using TF Layers) on TensorFlow
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
'''
# On command line: python3 logic_gate_linear_regressor.py
# Prerequisite: tensorflow 1.0 (see tensorflow.org)

from __future__ import print_function

import tensorflow as tf
import numpy as np
import tensorflow.contrib.learn as learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn
tf.logging.set_verbosity(tf.logging.INFO)

learning_rate = 0.01
# try other values for nhidden
nhidden = 16

def fnn_model_fn(features,labels,mode):
    print(features)
    print(labels)
    # output_labels = tf.reshape(labels,[-1,1])
    dense = tf.layers.dense(features,units=nhidden,activation=tf.nn.relu,use_bias=True)
    print(dense)
    logits = tf.layers.dense(dense,units=1,use_bias=True)
    print(logits)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=1)
    if mode != learn.ModeKeys.EVAL:
        # loss = tf.losses.sigmoid_cross_entropy(output_labels,logits)
        # loss = tf.losses.mean_squared_error(labels=output_labels,predictions=logits)
        loss = tf.losses.softmax_cross_entropy(
             onehot_labels=onehot_labels, logits=logits)
    if mode==learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=learning_rate,
            optimizer="SGD")
    predictions = {
        "classes": tf.round(logits),
        "probabilities": tf.nn.softmax(
             logits, name="softmax_tensor")
    }
    return model_fn.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(arg):
    x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    # try other logics; xor = [0., 1., 1., 0.], or = [0., 1., 1., 1.], and = [0., 0., 0., 1.], etc
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
    classifier = learn.Estimator(model_fn=fnn_model_fn, model_dir="/tmp/fnn")
    to_log = {"probabilities": "softmax_tensor"}
    log_hook = tf.train.LoggingTensorHook(to_log, every_n_iter=10)
    classifier.fit(x=x_data, y=y_data, batch_size=1, steps=50, monitors=[log_hook])
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    eval_results = classifier.evaluate(
        x=x_data, y=y_data, metrics=metrics)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()
