import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.initializers import RandomNormal
from keras.constraints import Constraint
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt

from layers.gumbel_softmax import GumbelSoftmaxActivation

def load(inputfile):
    data = pd.read_csv(inputfile).to_numpy()
    in_data = data[:,0:2]
    n_active = data[:,2].astype(int)
    first_active = data[:,3].astype(int)

    # scale input data (x, vx)
    weights = 1 / np.sqrt(in_data[:, 0] ** 2 + in_data[:, 1] ** 2)
    scaler = StandardScaler()
    scaler.fit(in_data, sample_weight=weights)
    in_data_scaled = scaler.transform(in_data)

    enc = OneHotEncoder(sparse=False)
    activations = np.concatenate((n_active.reshape((-1, 1)),
                                  first_active.reshape((-1, 1))), axis=1)
    enc.fit(activations)
    activations = enc.transform(activations)

    return np.hstack([in_data_scaled, activations]), enc

def get_discriminator_model(input_dim=221):
    # define model
    input_layer = Input(shape=input_dim)
    # data input
    in_data = Input(shape=2)

    critic_n = Concatenate()([input_layer[:, 0:5], in_data])
    critic_n = Dense(100)(critic_n)
    critic_n = LeakyReLU(alpha=0.2)(critic_n)
    critic_n = Dense(50)(critic_n)
    critic_n = LeakyReLU(alpha=0.2)(critic_n)
    critic_n = Dense(25)(critic_n)

    critic_w = Concatenate()([input_layer[:, 5:221], in_data])
    critic_w = Dense(400)(critic_w)
    critic_w = LeakyReLU(alpha=0.2)(critic_w)
    critic_w = Dense(200)(critic_w)
    critic_w = LeakyReLU(alpha=0.2)(critic_w)
    critic_w = Dense(100)(critic_w)
    critic_w = LeakyReLU(alpha=0.2)(critic_w)
    critic_w = Dense(50)(critic_w)
    critic_w = LeakyReLU(alpha=0.2)(critic_w)

    critic = Concatenate()([critic_n, critic_w])
    critic = Dense(4)(critic)
    critic = LeakyReLU(alpha=0.2)(critic)

    out_layer = Dense(1)(critic)
    # define model
    d_model = Model([input_layer, in_data], out_layer)
    return d_model

def get_generator_model(latent_dim):
    # noise input
    in_lat = Input(shape=latent_dim)
    # data input
    in_data = Input(shape=2)

    gen = Concatenate()([in_lat, in_data])
    gen = Dense(50)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(100)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(200)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(400)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Dense(400)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)

    n_active = gen[:, 0:50]
    n_active = Dense(5)(n_active)
    n_active = LeakyReLU(alpha=0.2)(n_active)
    n_active = GumbelSoftmaxActivation(n_categorical_variables=1, tau=0.2, name='GumbelSoftmax_N')(n_active)

    first_active = gen[:, 50:400]
    first_active = Dense(350)(first_active)
    first_active = LeakyReLU(alpha=0.2)(first_active)
    first_active = Dense(216)(first_active)
    first_active = LeakyReLU(alpha=0.2)(first_active)
    first_active = GumbelSoftmaxActivation(n_categorical_variables=1, tau=0.2, name='GumbelSoftmax_W')(first_active)
    # output
    out_layer = Concatenate()([n_active, first_active])
    # define model
    g_model = Model([in_lat, in_data], out_layer)
    return g_model

class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        total_n_samples,
        latent_dim,
        discriminator_extra_steps=3,
        gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.total_n_samples = total_n_samples
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        # Lists to keep track of loss
        self.d_loss_hist = list()
        self.g_loss_hist = list()

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images, in_data):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator([interpolated, in_data], training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            random_in_data = real_images[:,0:2]

            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator([random_latent_vectors, random_in_data], training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator([fake_images, random_in_data], training=True)
                # Get the logits for the real images
                real_logits = self.discriminator([real_images[:,2:], random_in_data], training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images[:,2:], fake_images, random_in_data)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_in_data = real_images[:,0:2]
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator([random_latent_vectors, random_in_data], training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator([generated_images, random_in_data], training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        # Keep track of losses
        #self.d_loss_hist.append(d_loss)
        #self.g_loss_hist.append(g_loss)
        return {"d_loss": d_loss, "g_loss": g_loss}

class GANMonitor(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 50 == 0:
            filename = 'generator_model_%03d.h5' % (epoch + 1)
            self.model.generator.save(filename)


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.i = 0
        self.x = []
        self.d_loss = []
        self.g_loss = []
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        self.logs.append(logs)
        self.x.append(self.i)
        self.d_loss.append(logs.get('d_loss'))
        self.g_loss.append(logs.get('g_loss'))
        self.i += 1

        plt.plot(self.x, self.d_loss, label="Critic loss")
        plt.plot(self.x, self.g_loss, label="Gen loss")
        plt.legend()
        plt.savefig('plot_line_plot_loss.png', dpi=400)
        plt.close()


class StepDecay():
    def __init__(self, initLearningRate=0.01, factor=0.25, dropEvery=10):
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initLearningRate = initLearningRate
        self.factor = factor
        self.dropEvery = dropEvery

    def __call__(self, epoch):
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        lr = self.initLearningRate * (self.factor ** exp)
        # return the learning rate
        return float(lr)


if __name__ == "__main__":

    # GPU memory usage configuration
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    backend.set_session(sess)

    BATCH_SIZE = 1024
    LATENT_DIM = 400
    EPOCHS = 3000
    LEARNING_RATE = 0.0005
    K = 5

    inputfile = "/home/ruben/GAN_muon_simulation/data/sim2.csv"

    print('_' * 100)
    print('OPTIONS WGAN-GP')
    print('_' * 100)
    print(' Training data file: {:<50}'.format(inputfile))
    print('     Training epochs: {:<50}'.format(EPOCHS))
    print('     Dimension of latent space: {:<50}'.format(LATENT_DIM))
    print('     Batch size: {:<50}'.format(BATCH_SIZE))
    print('     k hyperparameter: {:<50}'.format(K))
    print('_' * 100)

    print('Loading data: '+inputfile)
    train_samples, encoder = load(inputfile)
    print(f"Data shape: {train_samples.shape}")

    d_model = get_discriminator_model(input_dim=221)
    d_model.summary()
    g_model = get_generator_model(latent_dim=LATENT_DIM)
    g_model.summary()

    # Instantiate the optimizer for both networks
    # (learning_rate=0.0002, beta_1=0.5 are recommended)
    generator_optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=0.5, beta_2=0.9
    )

    # Define the loss functions for the discriminator,
    # which should be (fake_loss - real_loss).
    # We will add the gradient penalty later to this loss function.
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss

    # Define the loss functions for the generator.
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)

    # Instantiate the custom Keras callbacks.
    cbk = GANMonitor()
    pltloss = PlotLosses()
    lrs = LearningRateScheduler(StepDecay(initLearningRate=LEARNING_RATE, factor=0.75, dropEvery=100), verbose=0)


    # Instantiate the WGAN model.
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        total_n_samples=train_samples.shape[0],
        latent_dim=LATENT_DIM,
        discriminator_extra_steps=K,
    )

    # Compile the WGAN model.
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )

    # Start training the model.
    wgan.fit(train_samples, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cbk, pltloss, lrs])