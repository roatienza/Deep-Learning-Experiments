# Deep Learning Lecture Notes and Experiments
### Code samples have links to other repo that I maintain ([Advanced Deep Learning with Keras](https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras) book) or contribute ([Keras](https://github.com/keras-team/keras))

## 2022 Version
Welcome to the 2022 version of Deep Learning course. We made major changes in the coverage and delivery of this course to reflect the recent advances in the field.

What is new in 2022 version:

1) Emphasis on tools to use and deploy deep learning models. In the past, we learn how to build and train models to perform certain tasks. However, often times we want to use a pre-trained model for immediate deployment. testing or demonstration. Hence, we will use tools such as `huggingface`, `gradio` and `streamlit` in our discussions.

2) Emphasis on understanding deep learning building blocks. The ability to build, train and test models is important. However, when we want to optimize and deploy a deep learning model on a new hardware or run it on production, we need an in-depth understanding of the code implementation of our algorithms. Hence, there will be emphasis on low-level algorithms and their code implementations.

3) Emphasis on practical applications. Deep learning can do a lot more than recognition. Hence, we will highlight practical applications in vision (detection, segmentation), speech (ASR, TTS) and text (sentiment, summarization).

4) Various levels of abstraction. We will present deep learning concepts from low-level `numpy` and `einops`, to mid-level framework such as PyTorch, and to high-level APIs such as `huggingface`, `gradio` and `streamlit`. This enables us to use deep learning principles depending on the problem constraints.

5) Emphasis on individual presentation of assignments, machine exercises and projects. Online learning is hard. To maximize student learning, this course focuses on exchange of ideas to ensure individual student progress. 

### Coverage:
1. Deep Learning Toolkit - Anaconda, `venv`, VSCode, Python, Numpy, Einops, PyTorch, Timm, HuggingFace, Gradio, Streamlit, Colab, Deepnote, Kaggle, etc.
  - Overview: [PDF](versions/2022/overview/Overview.pdf), [YouTube](https://youtu.be/zU37kvvkz0o)
  - Development Environment and Code Editor: [PDF](versions/2022/tools/Toolkit_Env_Editor.pdf), [YouTube](https://youtu.be/LildU3tGGEo)
2. Datasets - collection, labelling, loading, splitting, feeding
3. Supervised Learning
4. Building blocks - MLPs, CNNs, RNNs, Transformers
5. Backpropagation, Optimization and Regularization
6. Unsupervised Learning
7. AutoEncoders and Variational AutoEncoders
8. Practical Applications


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
   
    
## Star, Fork, Cite
If you find this work useful, please give it a star, fork, or cite:

```
@misc{atienza2020dl,
  title={Deep Learning Lecture Notes},
  author={Atienza, Rowel},
  year={2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/roatienza/Deep-Learning-Experiments}},
}
```

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
- Keras Sample Code
  - [Simple Embedding and Sentiment Analysis](keras/embedding/sentiment_analysis.ipynb)
  - [Glove Embedding](keras/embedding/glove_embedding.ipynb)
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

