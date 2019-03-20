# -*- coding: utf-8 -*-
""" Simple implementation of Generative Adversarial Neural Network """
import os
import numpy as np

from IPython.core.debugger import Tracer

#from keras.datasets import mnist
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam

import matplotlib.pyplot as plt
plt.switch_backend('agg')   # allows code to run without a system DISPLAY


class GAN(object):
    """ Generative Adversarial Network class """
    def __init__(self, width=32, height=32, channels=3):

        self.width = width
        self.height = height
        self.channels = channels

        self.shape = (self.width, self.height, self.channels)

        self.optimizer = Adam(lr=0.0002, beta_1=0.5, decay=8e-8)

        self.G = self.__generator()
        self.G.compile(loss='binary_crossentropy', optimizer=self.optimizer)

        self.D = self.__discriminator()
        self.D.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])

        self.stacked_generator_discriminator = self.__stacked_generator_discriminator()

        self.stacked_generator_discriminator.compile(loss='binary_crossentropy', optimizer=self.optimizer)



    def __generator(self, z_dim=100):
        """ Declare generator """

        model = Sequential()

        # Hidden layer
        model.add(Dense(128, input_dim=z_dim))

        # Leaku ReLU
        model.add(LeakyReLU(alpha=0.01))

        # Output layer with tanh activation
        model.add(Dense(self.height*self.width*self.channels, activation="tanh"))
        model.add(Reshape(self.shape))

        return model

    def __discriminator(self):
        """ Declare discriminator """

        model = Sequential()

        model.add(Flatten(input_shape=self.shape))

        # Hidden layer
        model.add(Dense(128))

        # Leaku ReLU          
        model.add(LeakyReLU(alpha=0.01))
        # Output layer with sigmoid activation
        model.add(Dense(1, activation='sigmoid'))
        
        return model

    def __stacked_generator_discriminator(self):

        self.D.trainable = False

        model = Sequential()
        model.add(self.G)
        model.add(self.D)

        return model

    def train(self, X_train, epochs=200000, batch = 32, save_interval = 1000):

        for cnt in range(epochs):

            ## train discriminator
            random_index = np.random.randint(0, len(X_train) - np.int64(batch/2))
            legit_images = X_train[random_index : random_index + np.int64(batch/2)].reshape(np.int64(batch/2), self.width, self.height, self.channels)

            gen_noise = np.random.normal(0, 1, (np.int64(batch/2), 100))
            syntetic_images = self.G.predict(gen_noise)

            x_combined_batch = np.concatenate((legit_images, syntetic_images))
            y_combined_batch = np.concatenate((np.ones((np.int64(batch/2), 1)), np.zeros((np.int64(batch/2), 1))))

            d_loss = self.D.train_on_batch(x_combined_batch, y_combined_batch)


            # train generator

            noise = np.random.normal(0, 1, (batch, 100))
            y_mislabled = np.ones((batch, 1))

            g_loss = self.stacked_generator_discriminator.train_on_batch(noise, y_mislabled)

            print ('epoch: %d, [Discriminator :: d_loss: %f], [ Generator :: loss: %f]' % (cnt, d_loss[0], g_loss))

            if cnt % save_interval == 0:
                self.plot_images(save2file=True, step=cnt)


    def plot_images(self, save2file=False, samples=16, step=0):
        ''' Plot and generated images '''
        if not os.path.exists("./images"):
            os.makedirs("./images")
        filename = "./images/mnist_%d.png" % step
        noise = np.random.normal(0, 1, (samples, 100))

        images = self.G.predict(noise)

        plt.figure(figsize=(10, 10))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            try:
                image = np.reshape(image, [self.height, self.width, self.channels])
            except:
                image = np.reshape(image, [self.height, self.width])
            plt.imshow(image, cmap="gray")
            plt.axis('off')
        plt.tight_layout()

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    xl = x_train
    yl = y_train
    x_train = []
    for x, y in zip(xl, yl):
        if y == 3:
            x_train.append(x)

    x_train = np.array(x_train)
            
    # Rescale -1 to 1
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=3)

    gan = GAN(32, 32, 3)
    gan.train(x_train)
