'''
DCGAN on MNIST using Keras
Author: Rowel Atienza and Pablo Rodriguez Bertorello
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

# Library
import time
from matplotlib import pyplot as plt
import numpy as np

np.random.seed(123)

# Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

# Data
from tensorflow.examples.tutorials.mnist import input_data

class DiscriminatorGeneratorAdversarialNetwork(object):
    
    def __init__(self, img_rows=28, img_cols=28, channel=1):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        # discriminator
        discriminator_model = self.new_discriminator_model()  
        self.compiled_discriminator_model = self.new_compiled_discriminator_model(discriminator_model)  
        # adversary
        self.generator_model = self.new_generator_model()
        self.compiled_adversary_model = self.new_compiled_adversary_model(self.generator_model, discriminator_model)

    # (W−F+2P)/S+1
    def new_discriminator_model(self):
        sequence = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        sequence.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same'))
        sequence.add(LeakyReLU(alpha=0.2))
        sequence.add(Dropout(dropout))

        sequence.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        sequence.add(LeakyReLU(alpha=0.2))
        sequence.add(Dropout(dropout))

        sequence.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        sequence.add(LeakyReLU(alpha=0.2))
        sequence.add(Dropout(dropout))

        sequence.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        sequence.add(LeakyReLU(alpha=0.2))
        sequence.add(Dropout(dropout))

        # Out: 1-dim probability
        sequence.add(Flatten())
        sequence.add(Dense(1))
        sequence.add(Activation('sigmoid'))
        sequence.summary()
        return sequence

    
    def new_compiled_discriminator_model(self, discriminator_model):
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        sequence = Sequential()
        sequence.add(discriminator_model)
        sequence.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return sequence

    
    def new_generator_model(self):
        sequence = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        sequence.add(Dense(dim*dim*depth, input_dim=100))
        sequence.add(BatchNormalization(momentum=0.9))
        sequence.add(Activation('relu'))
        sequence.add(Reshape((dim, dim, depth)))
        sequence.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        sequence.add(UpSampling2D())
        sequence.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        sequence.add(BatchNormalization(momentum=0.9))
        sequence.add(Activation('relu'))

        sequence.add(UpSampling2D())
        sequence.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        sequence.add(BatchNormalization(momentum=0.9))
        sequence.add(Activation('relu'))

        sequence.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        sequence.add(BatchNormalization(momentum=0.9))
        sequence.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        sequence.add(Conv2DTranspose(1, 5, padding='same'))
        sequence.add(Activation('sigmoid'))
        sequence.summary()
        return sequence

    # The adversarial model is: generator, discriminator stacked together
    def new_compiled_adversary_model(self, generator_model, discriminator_model):
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        sequence = Sequential()
        sequence.add(generator_model)
        sequence.add(discriminator_model)
        sequence.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return sequence
    
    def get_generator_model(self):
        return self.generator_model
        
    def get_compiled_discriminator_model(self):
        return self.compiled_discriminator_model
    
    def get_compiled_adversary_model(self):
        return self.compiled_adversary_model

    
    class MNIST_GAN(object):
    
    def __init__(self, batch_size=256, save_interval=500, save2file):
        self.batch_size = batch_size
        # log
        self.save_interval = save_interval
        self.save2file = save2file
        # config
        self.channel = 1
        self.img_rows = 28
        self.img_cols = 28
        # true training set: load, shape, type 
        self.x_train = input_data.read_data_sets("mnist", one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)
        # instantiate multi-model network
        self.DCGAN = DiscriminatorGeneratorAdversarialNetwork()

        
    def train(self, train_steps=2000):
        for i in range(train_steps):
            # images: real 
            images_train = self.x_train[np.random.randint(0, self.x_train.shape[0], size=self.batch_size), :, :, :]
            
            # images: fake 
            noise = self.get_noise(self.batch_size)
            images_fake = self.DCGAN.get_generator_model().predict(noise)
            
            # images: real + fake
            x_train_plus_fake = np.concatenate((images_train, images_fake))
            
            # labels for true and fake images
            y_train_plus_fake = np.ones([2*self.batch_size, 1]) 
            y_train_plus_fake[self.batch_size:, :] = 0
            
            # train discriminator
            discriminator_loss = self.DCGAN.get_compiled_discriminator_model(). \
                train_on_batch(x_train_plus_fake, y_train_plus_fake)

            # train adversary
            y_fake = np.ones([self.batch_size, 1])
            noise = self.get_noise(self.batch_size)            
            adversary_loss = self.DCGAN.get_compiled_adversary_model(). \
                train_on_batch(noise, y_fake)
            
            self.print_log(i, adversary_loss, discriminator_loss)
            
            
    def get_noise(self, noise_size):
        return np.random.uniform(-1.0, 1.0, size=[noise_size, 100])

                    
    def print_log(self, i, adversary_loss, discriminator_loss):
        log_mesg = "%d:  [ADVERSARY_LOG loss: %f, fake_as_good_as_real accuracy: %f]" % (i, adversary_loss[0], adversary_loss[1])
        log_mesg = "%s [DISCRIMINATOR_LOG loss: %f, fake_detection accuracy: %f]" % (log_mesg, discriminator_loss[0], discriminator_loss[1])
        print(log_mesg)
        if (i+1)%self.save_interval==0:
            self.plot_step_images(step_number=(i+1)) # samples=noise_input.shape[0], 
            
            
    def plot_step_images(self, step_number):
        num_samples = 16
        # images: real
        i = np.random.randint(0, self.x_train.shape[0], num_samples)
        real_images = self.x_train[i, :, :, :]
        # images: generate fake
        noise = self.get_noise(num_samples)
        fake_images = self.DCGAN.get_generator_model().predict(noise)
        # plot
        real_filename = 'mnist.png'
        fake_filename = "mnist_%d.png" % step_number
        self.plot_images(real_images, real_filename)
        self.plot_images(fake_images, fake_filename)
        
    def plot_images(self, sample_images, file_name):
        folder_name = './images/'
        plt.figure(figsize=(10,10))
        for i in range(sample_images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = sample_images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        if self.save2file:
            plt.savefig(folder_name + file_name)
            plt.close('all')

            
if __name__ == '__main__':
    start_time = time.time()
    
    # instantiate & train
    mnist_dcgan = MNIST_GAN(save_interval=100, save2file=False) 
    mnist_dcgan.train(train_steps=10000)    

    end_time = time.time()
    print("Elapsed: %s seconds" % start_time - end_time)            
