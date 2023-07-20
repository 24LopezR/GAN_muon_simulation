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

def load(inputfile):
    data = pd.read_csv(inputfile).to_numpy()
    # Select only some radius
    mask = (data[:,8] == 16)
    data = data[mask]

    variables = data[:,:8]
    labels = data[:,8]

    # Scale and encode the variables
    scaler = StandardScaler()
    weights = 1/np.sqrt(data[:,4]**2 + data[:,5]**2)
    scaler.fit(variables, sample_weight=weights)
    variables = scaler.transform(variables)
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(labels.reshape((-1,1)))
    labels = encoder.transform(labels.reshape((-1,1)))

    return [variables, labels], scaler

def get_discriminator_model(in_shape):
    # first detector and label input
    in_first_det_data = Input(shape=in_shape)
    # second detector input
    in_second_detector = Input(shape=in_shape)
    # concat label as a channel
    merge = Concatenate()([in_second_detector, in_first_det_data])

    fe = Dense(64)(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    #fe = Dropout(0.4)(fe)
    fe = Dense(32)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    #fe = Dropout(0.4)(fe)
    fe = Dense(16)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    #fe = Dropout(0.4)(fe)

    # output
    out_layer = Dense(1, activation='linear')(fe)
    # define model
    d_model = Model([in_second_detector, in_first_det_data], out_layer)
    return d_model

def get_generator_model(in_shape, latent_dim):
    # data and label input
    in_first_detector = Input(shape=in_shape)
    # noise input
    in_lat = Input(shape=latent_dim)

    merge = Concatenate()([in_lat, in_first_detector])

    gen = Dense(16)(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    #gen = Dropout(0.3)(gen)
    gen = Dense(32)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    #gen = Dropout(0.3)(gen)
    gen = Dense(64)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    #gen = Dropout(0.3)(gen)

    # output
    out_layer = Dense(4, activation='linear')(gen)
    # define model
    g_model = Model([in_lat, in_first_detector], out_layer)
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

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, real_data):
        if isinstance(real_data, tuple):
            real_data = real_data[0]
            if len(real_data) == 2:
                sample_weight = real_data[0]
            else:
                sample_weight = None


        # Get the batch size
        batch_size = tf.shape(real_data)[0]

        # Split dataset
        real_samples = real_data[:, 4:8]
        in_data = real_data[:, 0:4]

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            # Decode them to fake images
            fake_samples = self.generator([random_latent_vectors, in_data], training=True)

            # Combine them with real images
            combined_samples = tf.concat([fake_samples, real_samples], axis=0)
            combined_in_data = tf.concat([in_data, in_data], axis=0)

            # Assemble labels discriminating real from fake images
            labels = tf.concat(
                [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
            )
            # Add random noise to the labels - important trick!
            labels += 0.05 * tf.random.uniform(tf.shape(labels))

            # Train the discriminator
            with tf.GradientTape() as tape:
                predictions = self.discriminator([combined_samples, combined_in_data], training=True)
                d_loss = self.loss_fn(labels, predictions)
            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights)
            )

        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            generated_samples = self.generator([random_latent_vectors, in_data], training=True)
            predictions = self.discriminator([generated_samples, in_data], training=True)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Keep track of losses
        return {"d_loss": d_loss, "g_loss": g_loss}

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, saveEvery=50):
        self.saveEvery = saveEvery

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.saveEvery == 0:
            filename = 'generator_model_%03d.h5' % (epoch + 1)
            self.model.generator.save(filename)


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.x = []
        self.d_loss = []
        self.g_loss = []
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(epoch)
        self.d_loss.append(logs['d_loss'])
        self.g_loss.append(logs['g_loss'])

        plt.plot(self.x, self.d_loss, label="Critic loss", linewidth=0.5)
        plt.plot(self.x, self.g_loss, label="Gen loss", linewidth=0.5)
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
    LATENT_DIM = 16
    EPOCHS = 1000
    LEARNING_RATE = 0.001
    K = 1

    inputfile = "/home/ruben/fewSamples/training_samples.csv"

    print('_' * 100)
    print('OPTIONS LScGAN')
    print('_' * 100)
    print(' Training data file: {:<50}'.format(inputfile))
    print('     Training epochs: {:<50}'.format(EPOCHS))
    print('     Dimension of latent space: {:<50}'.format(LATENT_DIM))
    print('     Batch size: {:<50}'.format(BATCH_SIZE))
    print('     k hyperparameter: {:<50}'.format(K))
    print('_' * 100)

    print('Loading data...')
    dataset, scaler = load(inputfile)
    variables, labels = dataset

    num_classes = labels.shape[1]

    d_model = get_discriminator_model(in_shape=4)
    d_model.summary()
    g_model = get_generator_model(in_shape=4, latent_dim=LATENT_DIM)
    g_model.summary()

    # Instantiate the optimizer for both networks
    # (learning_rate=0.0002, beta_1=0.5 are recommended)
    generator_optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE, beta_1=0.5, beta_2=0.9
    )

    # Instantiate the custom Keras callbacks.
    cbk = GANMonitor(saveEvery=25)
    plot_losses = PlotLosses()
    lrs = LearningRateScheduler(StepDecay(initLearningRate=LEARNING_RATE, factor=0.5, dropEvery=200), verbose=0)

    # Instantiate the WGAN model.
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        total_n_samples=variables.shape[0],
        latent_dim=LATENT_DIM,
        discriminator_extra_steps=K,
    )

    # Compile the WGAN model.
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        loss_fn=tf.losses.MeanSquaredError(name='mse')
    )

    # Start training the model.
    wgan.fit(variables, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cbk, plot_losses, lrs])