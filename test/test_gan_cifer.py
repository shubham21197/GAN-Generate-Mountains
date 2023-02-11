from __future__ import print_function, division

from keras.datasets import cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.layers.convolutional import Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        model = Sequential()
        model.add(self.generator)
        self.discriminator.trainable = False
        model.add(self.discriminator)

        model.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.combined = model

        # The generator takes noise as input and generated imgs
        # z = Input(shape=(1024,))
        # img = self.generator(z)

        # # For the combined model we will only train the generator
        # self.discriminator.trainable = False

        # # The valid takes generated images as input and determines validity
        # valid = self.discriminator(img)

        # # The combined model  (stacked generator and discriminator) takes
        # # noise as input => generates images => determines validity
        # self.combined = Model(z, valid)
        # self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def generate_noise(self, size):
        # generate points in the latent space
        x_input = np.random.randn(100 * size)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(size, 100)
        return x_input

    def build_generator(self):

        model = Sequential()

        n_nodes = 128 * 8 * 8
        model.add(Dense(n_nodes, input_dim=100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((8, 8, 128)))
        # upsample to 14x14
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 28x28
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

        model.summary()

        # noise = Input(shape=noise_shape)
        # img = model(noise)

        return model

    def build_discriminator(self):

        img_shape = (self.img_rows, self.img_cols, self.channels)

        model = Sequential()

        # model.add(Flatten(input_shape=img_shape))
        # model.add(Dense(512))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(256))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(1, activation='sigmoid'))

        model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        # img = Input(shape=img_shape)
        # validity = model(img)

        return model

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        (X_train, y_train), (_, _) = cifar10.load_data()

        y_train_flat = y_train.flatten()

        X_train = X_train[y_train_flat == 8]

        print(X_train.shape)


        # Rescale -1 to 1
        X_train = (X_train.astype('float32') - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=-1)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs + 1):

            # ---------------------
            #  Train Discriminator
            # ---------------------
            print("Training Discriminator: ", epoch)

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = self.generate_noise(half_batch)

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------
            print("Training Generator: ", epoch)

            noise = self.generate_noise(batch_size)

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss_fake[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch, X_train)

    def save_imgs(self, epoch, X_train = []):
        r, c = 5, 5
        noise = self.generate_noise(r * c)
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./images/mnist_%d.png" % epoch)
        plt.close()

        r, c = 5, 5
        noise = self.generate_noise(r * c)
        idx = np.random.randint(0, X_train.shape[0], 30)
        gen_imgs = X_train[idx]

        print(gen_imgs.shape)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                print(i, j, cnt)
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("./images/og_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=3000, batch_size=64, save_interval=100)