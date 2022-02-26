# Deep Learning Lecture Notes and Experiments

## [2022 Version](../README.md)

## [2020 Version](2020/README.md)

## Lecture Notes (Old - will no longer be maintained)
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
  - [Backprop on a single unit MLP](2020/backprop/backprop.ipynb)
- Keras Sample Code
  - [Overview](https://docs.google.com/presentation/d/15Y1snbE73g8vw16RN6uehVHyDFxAK_b0iKcmId1j5qM/edit?usp=sharing)
  - [MLP on Linear Model](keras/mlp/linear.ipynb)
  - [MNIST Sampler](2020/keras/mlp/mnist-sampler.ipynb)
  - [MLP on MNIST](2020/keras/mlp/mlp-mnist.ipynb)
4. [Regularization](https://docs.google.com/presentation/d/1lg4oxRDvfUIEtzMJ7E-Lqv1cDNiwoNeT1r5T-XnFIQI/edit?usp=sharing)
- Keras Sample Code
  - [MLP on MNIST no Regularizer](2020/keras/regularization/mlp-mnist-noreg.ipynb)
  - [MLP on MNIST with L2](2020/keras/regularization/mlp-mnist-l2.ipynb)
  - [MLP on MNIST with Dropout](2020/keras/regularization/mlp-mnist-dropout.ipynb)
  - [MLP on MNIST with Data Augmentation](2020/keras/regularization/mlp-mnist-data_augment.ipynb) 
  
5. [Optimization](https://docs.google.com/presentation/d/1wt53ds5dywq3WUm-jkdKFUjiHayBAV6-CSFAJg76Clg/edit?usp=sharing)
  
6. [Convolutional Neural Networks (CNN)](https://docs.google.com/presentation/d/1vxCMwjbssYKisIWt2UYiuOFMsJaFv-5-I6mYvtJ6Hr8/edit?usp=sharing)
- Keras Sample Code
  - [CNN on MNIST](2020/keras/cnn/cnn-mnist.ipynb)
  - [CNN on MNIST using Functional API](2020/keras/cnn/cnn-functional.ipynb)
  - [CNN on MNIST Siamese Network](2020/keras/cnn/cnn-siamese.ipynb)
  
7. [Deep Networks](https://docs.google.com/presentation/d/14aFawAa4zNqvPRkhmS5YATVxSlS01fig7qnstecRgG0/edit?usp=sharing)
- Keras Sample Code
  - [DenseNet](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/densenet-cifar10-2.4.1.py)  
  - [ResNet](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
8. [Embeddings](https://docs.google.com/presentation/d/1YtKWA53T2NqXoL0vnk8jWl1WCBWCLhbh5OWz_1JrGdU/edit?usp=sharing) 
- Keras Sample Code
  - [Simple Embedding and Sentiment Analysis](../keras/embedding/sentiment_analysis.ipynb)
  - [Glove Embedding](2020/keras/embedding/glove_embedding.ipynb)
9. [Recurrent Neural Networks, LSTM, GRU](https://docs.google.com/presentation/d/1qjQkUwnr2V--7JPz0H_wkzRyTYX3UtJsYrB3MQPGKLE/edit?usp=sharing)
- Keras Sample Code
  - [SimpleRNN on MNIST](2020/keras/rnn/simple-rnn-mnist.ipynb)
  - [CuDNNLSTM on MNIST](2020/keras/rnn/cudnnlstm-mnist.ipynb)
10. [AutoEncoders](https://docs.google.com/presentation/d/1gXWl0luuDe1qoQLSKdOUrzoq51WhhNLeq7x3PxQXYkA/edit?usp=sharing)
- Keras Sample Code
  - [AutoEncoder](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/autoencoder-mnist-3.2.1.py)
  - [Denoising AutoEncoder](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/denoising-autoencoder-mnist-3.3.1.py)
  - [Colorization AutoEncoder](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/colorization-autoencoder-cifar10-3.4.1.py)
11. [Generative Adversarial Networks (GAN)](https://docs.google.com/presentation/d/13fiFibqjl9ps_CktJzMNAvoZXOlzHQDu8eRSb3a227g/edit?usp=sharing)
- Keras Sample Code
  - [DCGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter4-gan/dcgan-mnist-4.2.1.py)
  - [CGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter4-gan/cgan-mnist-4.3.1.py)
  
11a. [Improved GANs](https://docs.google.com/presentation/d/1tATpY1gzJo8x6Ziceln-KjcgEeqX-fnapL6IFdc9Wbk/edit?usp=sharing)
- Keras Sample Code
  - [WGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter5-improved-gan/wgan-mnist-5.1.2.py)
  - [LSGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter5-improved-gan/lsgan-mnist-5.2.1.py)
  - [ACGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter5-improved-gan/acgan-mnist-5.3.1.py)
  
11b. [Disentangled GAN](https://docs.google.com/presentation/d/1XboUGxLB1wYqJppsYiToq120JLhAOiuxLMUNbSfy_dk/edit?usp=sharing)
  - [InfoGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter6-disentangled-gan/infogan-mnist-6.1.1.py)
  
11c. [Cross-Domain GAN](https://docs.google.com/presentation/d/17lizm6BGtDB7OIR1XQHGH-VzcoM5N0vBnjOhIlcAwZs/edit?usp=sharing)
  - [CycleGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter7-cross-domain-gan/cyclegan-7.1.1.py)

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
14. [Policy Gradient Methods](https://docs.google.com/presentation/d/1SsPomQARNVKuIW4UtsLYkylM1b_ZuTIySCZSdEkmGTM/edit?usp=sharing)
- Keras Sample Code
  - [REINFORCE. REINFORCE w/ Baseline, Actor-Critic, and A2C](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter10-policy/policygradient-car-10.1.1.py)  

