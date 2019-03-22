# Deep Learning Lecture Notes and Experiments
### Code samples have links to other repo that I maintain ([Advanced Deep Learning with Keras](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras) book) or contribute ([Keras](https://github.com/keras-team/keras))
## Lecture Notes
0. Course Roadmap
  - [Deep Learning](https://docs.google.com/presentation/d/1JSmMV3SICXJ3SQOSj5WgJipssOpBYZKNZm1YnU5HO3g/edit?usp=sharing)
  - [Limits of Deep Learning](https://docs.google.com/presentation/d/13nsjiEjpiUpidxThT6hoCg19o0TMG-ho4tzgvuPkWV8/edit?usp=sharing)

1. Background Materials
  - [Linear Algebra](https://docs.google.com/presentation/d/1VBAb1C07dAQxRPIat0lDfTxWO9amnJvuHziNyVLkaFo/edit?usp=sharing)
  - [Probability](https://docs.google.com/presentation/d/1eygyQDzngXLGJjixNidUXJL-c12SFkQhztbdXDAhgR8/edit?usp=sharing)
  - [Numerical Computation](https://docs.google.com/presentation/d/1Nt9vr-8_Tcgez8jmz7sPsbXc0PDpU0d04-f2MFR5XZg/edit?usp=sharing)
2. Machine Learning Basics
  - [Concepts, Capacity, Estimators, Linear Regression](https://docs.google.com/presentation/d/1Xn6FaPiGTLnCRKQjIXySxpdteLVT1F8WDFketIUiTlI/edit?usp=sharing) 
  - [MLE, Bayesian, Other ML Algorithms](https://docs.google.com/presentation/d/1Dp2IBWnxQmKMszX0uL5HSfK302q6kpMfiexYAoT9z-k/edit?usp=sharing)
  - [Stochastic Gradient Descent, etc](https://docs.google.com/presentation/d/1Ss2BhwyarFGFiEIgqbQ-CW9zq0LJxmaOcHZKMzRWJ5k/edit?usp=sharing)
3. Deep Neural Networks
  - [Deep Feedforward Neural Networks, Cost, Output, Hidden Units](https://docs.google.com/presentation/d/1woHBsNgnwzjJndMcXXznaBKlLvWywuA6T7BFi0K7Yhg/edit?usp=sharing)
  - [Back Propagation](https://docs.google.com/presentation/d/1XD0tA6oxOETfFn1DTGJByhhyH3MF586OCN06WvAP22E/edit?usp=sharing)
- PyTorch Sample Code
  - [Backprop on a single unit MLP](backprop/backprop.ipynb)
- Keras Sample Code
  - [Overview](https://docs.google.com/presentation/d/15Y1snbE73g8vw16RN6uehVHyDFxAK_b0iKcmId1j5qM/edit?usp=sharing)
  - [MLP on Linear Model](keras/mlp/linear.ipynb)
  - [MNIST Sampler](keras/mlp/mnist-sampler.ipynb)
  - [MLP on MNIST](keras/mlp/mlp-mnist.ipynb)
4. [Regularization](https://docs.google.com/presentation/d/1lg4oxRDvfUIEtzMJ7E-Lqv1cDNiwoNeT1r5T-XnFIQI/edit?usp=sharing)
- Keras Sample Code
  - [MLP on MNIST no Regularizer](keras/regularization/mlp-mnist-noreg.ipynb)
  - [MLP on MNIST with L2](keras/regularization/mlp-mnist-l2.ipynb)
  - [MLP on MNIST with Dropout](keras/regularization/mlp-mnist-dropout.ipynb)
  - [MLP on MNIST with Data Augmentation](keras/regularization/mlp-mnist-data_augment.ipynb) 
  
5. [Optimization](https://docs.google.com/presentation/d/1wt53ds5dywq3WUm-jkdKFUjiHayBAV6-CSFAJg76Clg/edit?usp=sharing)
  
6. [Convolutional Neural Networks (CNN)](https://docs.google.com/presentation/d/1vxCMwjbssYKisIWt2UYiuOFMsJaFv-5-I6mYvtJ6Hr8/edit?usp=sharing)
- Keras Sample Code
  - [CNN on MNIST](keras/cnn/cnn-mnist.ipynb)
  - [CNN on MNIST using Functional API](keras/cnn/cnn-functional.ipynb)
  - [CNN on MNIST Siamese Network](keras/cnn/cnn-siamese.ipynb)
  
7. [Deep Networks](https://docs.google.com/presentation/d/14aFawAa4zNqvPRkhmS5YATVxSlS01fig7qnstecRgG0/edit?usp=sharing)
- Keras Sample Code
  - [DenseNet](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/densenet-cifar10-2.4.1.py)  
  - [ResNet](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
8. [Embeddings](https://docs.google.com/presentation/d/1YtKWA53T2NqXoL0vnk8jWl1WCBWCLhbh5OWz_1JrGdU/edit?usp=sharing) 
9. [Recurrent Neural Networks, LSTM, GRU](https://docs.google.com/presentation/d/1qjQkUwnr2V--7JPz0H_wkzRyTYX3UtJsYrB3MQPGKLE/edit?usp=sharing)
- Keras Sample Code
  - [SimpleRNN on MNIST](keras/rnn/simple-rnn-mnist.ipynb)
  - [CuDNNLSTM on MNIST](keras/rnn/cudnnlstm-mnist.ipynb)
10. [AutoEncoders](https://docs.google.com/presentation/d/1gXWl0luuDe1qoQLSKdOUrzoq51WhhNLeq7x3PxQXYkA/edit?usp=sharing)
- Keras Sample Code
  - [AutoEncoder](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/autoencoder-mnist-3.2.1.py)
  - [Denoising AutoEncoder](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/denoising-autoencoder-mnist-3.3.1.py)
  - [Colorization AutoEncoder](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/colorization-autoencoder-cifar10-3.4.1.py)
11. [Generative Adversarial Networks (GAN)](https://docs.google.com/presentation/d/13fiFibqjl9ps_CktJzMNAvoZXOlzHQDu8eRSb3a227g/edit?usp=sharing)
- Keras Sample Code
  - [DCGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter4-gan/dcgan-mnist-4.2.1.py)
  - [CGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter4-gan/cgan-mnist-4.3.1.py)
12. [Variational Autoencoder (VAE)](https://docs.google.com/presentation/d/1ORVwhh5PgWEehcUQYL9t8nBCk9cj4TgXmUJIl6WMkpo/edit?usp=sharing)
- Keras Sample Code
  - [VAE MLP](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/vae-mlp-mnist-8.1.1.py)
  - [VAE CNN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/vae-cnn-mnist-8.1.2.py)
  - [CVAE](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/cvae-cnn-mnist-8.2.1.py)
13. [Deep Reinforcement Learning (DRL)](https://docs.google.com/presentation/d/1oZC5qGofbx-dlPcnr00_fPBK0GEtt6FnMdpsnA6XYX8/edit?usp=sharing)
- Keras Sample Code
  - [Q-Learning](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/q-learning-9.3.1.py)
  - [Q-Learning on FrozenLake-v0](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/q-frozenlake-9.5.1.py)
  - [DQN and DDQN on CartPole-v0](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/dqn-cartpole-9.6.1.py)

### Warning: The following are old experiments that are longer updated and maintained
### [Tensorflow](https://www.tensorflow.org/) Experiments
1. [Hello World!](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Intro/hello.py) 
2. [Linear Algebra](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Math/linear_algebra.py)
3. [Matrix Decomposition](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Math/decomposition.py)
4. [Probability Distributions using TensorBoard](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Probability/distributions.py)
5. [Linear Regression by PseudoInverse](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Regression/linear_inv.py)
6. [Linear Regression by Gradient Descent](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Regression/linear_regression.py)
6. [Under Fitting in Linear Regression](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Machine_Learning/underfit_regression.py)
7. [Optimal Fitting in Linear Regression](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Machine_Learning/optfit_regression.py)
8. [Over Fitting in Linear Regression](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Machine_Learning/overfit_regression.py)
9. [Nearest Neighbor](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Machine_Learning/regression_nn.py)
10. [Principal Component Analysis](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Machine_Learning/pca.py)
11. [Logical Ops by a 2-layer NN (MSE)](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Neural_Networks/logic_gate_mse.py)
12. [Logical Ops by a 2-layer NN (Cross Entropy)](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Neural_Networks/logic_gate_logits.py)
13. NotMNIST Deep Feedforward Network: [Code for NN](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Deep_Networks/mnist_a2j_mlp.py) and [Code for Pickle](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Deep_Networks/mnist_a2j_2pickle.py)
14. [NotMNIST CNN](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Deep_Networks/mnist_a2j_cnn.py)
15. [word2vec](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Word2Vec/word2vec.py)
16. [Word Prediction/Story Generation using LSTM](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/rnn_words.py). Belling the Cat by Aesop [Sample Text Story](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/belling_the_cat.txt)

## [Keras](https://keras.io) on Tensorflow Experiments
1. [NotMNIST Deep Feedforward Network](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Deep_Networks/mnist_a2j_mlp_keras.py)
2. [NotMNIST CNN](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/Deep_Networks/mnist_a2j_cnn_keras.py)
3. [DCGAN on MNIST](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py)
