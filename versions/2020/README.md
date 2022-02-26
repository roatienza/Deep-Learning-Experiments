# Deep Learning Lecture Notes and Experiments
### Code samples have links to other repo that I maintain ([Advanced Deep Learning with Keras](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras) book) or contribute ([Keras](https://github.com/keras-team/keras))


## 2020 Version

So much have changed since this course was offerred. Hence, it is time to revise. I will keep the original lecture notes at the bottom. They will no longer be maintained. I am introducing 2020 version. Big changes that will happen are as follows:

1) Review of Machine Learning - Frustrated with the lack of depth in the ML part, I decided to develop a new course - [Foundations of Machine Learning](https://github.com/roatienza/ml). Before studying DL, a good grasp of ML is of paramount importance. Without ML, it is harder to understand DL and to move it forward.

2) Lecture Notes w/ Less Clutter - Prior to this version, my lecture notes have too much text. In the 2020 version, I am trying to focus more on the key concepts while carefully explaining during lecture the idea behind these concepts. The lecture notes are closely coupled with sample implementations. This enables us to quickly move from concepts to actual code implementations.

## Lecture Notes and Experiments

0. Course Roadmap
  - [Deep Learning](https://docs.google.com/presentation/d/1JSmMV3SICXJ3SQOSj5WgJipssOpBYZKNZm1YnU5HO3g/edit?usp=sharing)
  - [Limits of Deep Learning](https://docs.google.com/presentation/d/13nsjiEjpiUpidxThT6hoCg19o0TMG-ho4tzgvuPkWV8/edit?usp=sharing)
  
1. Multilayer Perceptron (MLP)
  - [Lecture Notes](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/MLP/MLP.pdf)
  - Experiments:
    - [Linear Regression](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/MLP/code/tf.keras/linear.ipynb)
    - [MNIST Sampler](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/MLP/code/tf.keras/mnist-sampler.ipynb)
    - [MNIST Classifier](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/MLP/code/tf.keras/mlp-mnist.ipynb)

2. Convolutional Neural Network (CNN)
  - [Lecture Notes](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/cnn/CNN.pdf)
  - Experiments:
    - [CNN MNIST Classifier](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/cnn/code/cnn-mnist.ipynb)
    - [CNN MNIST Classifier using Functional API](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/cnn/code/cnn-functional.ipynb)
    - [Y Network using Functional API](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/cnn/code/cnn-siamese.ipynb)
  - Deep CNN
    - [ResNet](https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py)
    - [DenseNet](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter2-deep-networks/densenet-cifar10-2.4.1.py)  
   
 3. Recurrent Neural Network (RNN)
  - [Lecture Notes](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/rnn/RNN.pdf)
  - Experiments:
    - [RNN MNIST Classifier](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/rnn/code/simple-rnn-mnist.ipynb)
    - [LSTM MNIST Classifier](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/rnn/code/cudnnlstm-mnist.ipynb)
    
 4. Transformer
  - [Lecture Notes](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/transformer/Transformer.pdf)
  - Experiments:
    - [Transformer MNIST Classifier](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/transformer/code)
  
 5. Regularizer
   - [Lecture Notes](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/regularizer/Regularizer.pdf)
   - Experiments:
     - [MLP on MNIST no Regularizer](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/regularizer/code/mlp-mnist-noreg.ipynb)
     - [MLP on MNIST with L2](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/regularizer/code/mlp-mnist-l2.ipynb)
     - [MLP on MNIST with Dropout](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/regularizer/code/mlp-mnist-dropout.ipynb)
     - [MLP on MNIST with Data Augmentation](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/regularizer/code/mlp-mnist-data_augment.ipynb) 
     
  5. Optimizer
   - [Lecture Notes](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/optimizer/Optimizer.pdf)
   
  6. AutoEncoder
   - [Lecture Notes](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/autoencoder/AutoEncoders.pdf)
   - Experiments:
     - [AutoEncoder](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/autoencoder-mnist-3.2.1.py)
     - [Denoising AutoEncoder](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/denoising-autoencoder-mnist-3.3.1.py)
     - [Colorization AutoEncoder](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter3-autoencoders/colorization-autoencoder-cifar10-3.4.1.py)
  
  7. Normalization
   - [Lecture Notes](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/normalization/Normalization.pdf)
    
  8. Generative Adversarial Network (GAN)
   - [GAN](https://docs.google.com/presentation/d/13fiFibqjl9ps_CktJzMNAvoZXOlzHQDu8eRSb3a227g/edit?usp=sharing)
   - [Improved GAN](https://docs.google.com/presentation/d/1tATpY1gzJo8x6Ziceln-KjcgEeqX-fnapL6IFdc9Wbk/edit?usp=sharing)
   - [Disentangled GAN](https://docs.google.com/presentation/d/1XboUGxLB1wYqJppsYiToq120JLhAOiuxLMUNbSfy_dk/edit?usp=sharing)
   - [Cross-Domain GAN](https://docs.google.com/presentation/d/17lizm6BGtDB7OIR1XQHGH-VzcoM5N0vBnjOhIlcAwZs/edit?usp=sharing)
   - Experiments:
     [DCGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter4-gan/dcgan-mnist-4.2.1.py)
           , [CGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter4-gan/cgan-mnist-4.3.1.py)
           , [WGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter5-improved-gan/wgan-mnist-5.1.2.py)
           , [LSGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter5-improved-gan/lsgan-mnist-5.2.1.py)
           , [ACGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter5-improved-gan/acgan-mnist-5.3.1.py)
           , [InfoGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter6-disentangled-gan/infogan-mnist-6.1.1.py)
           , [CycleGAN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter7-cross-domain-gan/cyclegan-7.1.1.py)
  
  9. Variational AutoEncoder (VAE) 
   - [Lecture Notes](https://docs.google.com/presentation/d/1ORVwhh5PgWEehcUQYL9t8nBCk9cj4TgXmUJIl6WMkpo/edit?usp=sharing)
   - Experiments:
     - [VAE MLP](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/vae-mlp-mnist-8.1.1.py)
     - [VAE CNN](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/vae-cnn-mnist-8.1.2.py)
     - [CVAE](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter8-vae/cvae-cnn-mnist-8.2.1.py)
  
  10. Object Detection 
   - [Lecture Notes](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/object_detection/Object_Detection.pdf)
   - Experiments:
     - [SSD](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/tree/master/chapter11-detection)
     
  11. Object Segmentation 
   - [Lecture Notes](https://github.com/roatienza/Deep-Learning-Experiments/blob/master/versions/2020/segmentation/Segmentation.pdf)
   - Experiments:
     - [FCN w/ PSPNet](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/tree/master/chapter12-segmentation)
 
  12. [Deep Reinforcement Learning (DRL)](https://docs.google.com/presentation/d/1oZC5qGofbx-dlPcnr00_fPBK0GEtt6FnMdpsnA6XYX8/edit?usp=sharing)
   - Experiments:
     - [Q-Learning](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/q-learning-9.3.1.py)
     - [Q-Learning on FrozenLake-v0](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/q-frozenlake-9.5.1.py)
     - [DQN and DDQN on CartPole-v0](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter9-drl/dqn-cartpole-9.6.1.py)

  13. [Policy Gradient Methods](https://docs.google.com/presentation/d/1SsPomQARNVKuIW4UtsLYkylM1b_ZuTIySCZSdEkmGTM/edit?usp=sharing)
   - Experiments:
     - [REINFORCE. REINFORCE w/ Baseline, Actor-Critic, and A2C](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/blob/master/chapter10-policy/policygradient-car-10.1.1.py)  
   
    
