import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import glob
import keras.backend as K

tf.config.threading.set_inter_op_parallelism_threads(4)

def minibatch_std_layer(layer, group_size=4):

    group_size = K.minimum(4, layer.shape[0])
    shape = layer.shape

    minibatch = K.reshape(layer,(group_size, -1, shape[1], shape[2]))
    minibatch -= tf.reduce_mean(minibatch, axis=0, keepdims=True)
    minibatch = tf.reduce_mean(K.square(minibatch), axis = 0)
    minibatch = K.square(minibatch + 1e-8) #epsilon=1e-8
    minibatch = tf.reduce_mean(minibatch, axis=[1,2], keepdims=True)
    minibatch = K.tile(minibatch,[group_size, 1, shape[2]])
    return K.concatenate([layer, minibatch], axis=1)

# The discriminator model takes an image as input and returns a probability
# indicating whether the image is real or fake
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[256, 256, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))
  
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.25))

    model.add(layers.Lambda(minibatch_std_layer))
  
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
  
    return model

# The generator model takes a noise vector as input and returns an image
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256) # Note: None is the batch size

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256, kernel_size = 3, padding = "same"))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256, kernel_size = 3, padding = "same"))
    assert model.output_shape == (None, 32, 32, 256)
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())
    
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(128, kernel_size = 3, padding = "same"))
    assert model.output_shape == (None, 64, 64, 128)
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())

    model.add(layers.UpSampling2D(size = (4, 4)))
    model.add(layers.Conv2D(128, kernel_size = 3, padding = "same"))
    assert model.output_shape == (None, 256, 256, 128)
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(3, kernel_size = 3, padding = "same"))
    model.add(layers.Activation("tanh"))

    return model



# Create the discriminator and generator models
discriminator = make_discriminator_model()
generator = make_generator_model()

# The generator takes noise as input and generates images
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

# Use Binary Crossentropy loss for both the generator and discriminator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Define the optimizers for both generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Define the training loop for the GAN
EPOCHS = 50
noise_dim = 100
num_examples_to_generate = 16
BATCH_SIZE = 32

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
  
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
      
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
        
        print("Generator loss: ", gen_loss)
        print("Discriminator loss: ", disc_loss)
      
        # Produce images for the GIF as we go
        # clear_output(wait=False)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed)
        
    # Generate after the final epoch
    # clear_output(wait=True)
        # Generate after the final epoch
    generate_and_save_images(generator,
                             epochs,
                             seed)

def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :] * 0.5 + 0.5)
        plt.axis('off')
        
    plt.savefig('genned/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()

def load_image(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [256, 256])
    image = (image - 127.5) / 127.5
    return image

def load_dataset(folder_path):
    all_images = glob.glob(folder_path + '/*.jpg')
    dataset = tf.data.Dataset.from_tensor_slices(all_images)
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

train_dataset = load_dataset('data/himalaya/compressed')
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Train the GAN
train(train_dataset, EPOCHS)

