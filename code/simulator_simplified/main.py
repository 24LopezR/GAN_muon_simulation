##################################################################################################
#### Executable script to train and evaluate a GAN                                            ####
#### that will simulate the Detector part                                                     ####
##################################################################################################
import importlib
import os
from optparse import OptionParser

import tensorflow as tf

import utils.cfg as conf
from utils.dataloader import load, scale
from utils.evaluating import Evaluation
from utils.generating_WGAN import generate_evaluation_samples
from utils.training_WGAN import train

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    def_inputfile = '/home/ruben/GAN_muon_simulation/data/sim2.csv'
    # _________________________________________________________________________________________
    # Read user options
    parser = OptionParser(usage="%prog --help")
    parser.add_option("-g", "--gan", dest="gan_model", type="string", default='WGAN', help="GAN model architecture")
    parser.add_option("-i", "--input", dest="input", type="string", default=def_inputfile, help="Input root filename")
    parser.add_option("-o", "--output", dest="output", type="string", default='output', help="Output filename")
    parser.add_option("-e", "--epochs", dest="epochs", type="int", default=200, help="N epochs")
    parser.add_option("-l", "--latent", dest="latent", type="int", default=100, help="input_dim of latent space")
    parser.add_option("-b", "--batch", dest="batch", type="int", default=256, help="N batch")
    parser.add_option("-k", "--khyp", dest="k_hyp", type="int", default=5, help="k hyperparameter")
    parser.add_option("-u", "--unrolling", dest="unrolling", type="int", default=1, help="unrolling hyperparameter")
    parser.add_option("-d", "--decode", dest="decode_size", type="int", default=1, help="Size for encoding dataset")
    parser.add_option("-r", "--learning-rate", dest="lr", type="float", default=0.00005, help="Learning rate")
    parser.add_option("-t", "--temperature", dest="tau", type="float", default=0.01, help="GS temperature")
    parser.add_option("-c", "--clip", dest="clip", type="float", default=0.01, help="Weight clipping")
    parser.add_option("-m", "--init", dest="init", type="string", default=None, help="Initial generator model")
    parser.add_option("-n", "--no-train", dest="train", action="store_false")
    (options, args) = parser.parse_args()

    # GAN model
    gan_m = options.gan_model
    conf.CONDITIONAL = True
    # Name of input file
    inputfile = options.input
    # Name of output file
    output = options.output
    # number of epochs
    epochs = options.EPOCHS
    # size of the latent space
    latent_dim = options.latent
    # batch size
    nbatch = options.batch
    # number or discriminator updates
    n_discrim_updates = options.k_hyp
    # number or generator updates
    n_gen_updates = options.unrolling
    # learning rate
    learning_rate = options.lr
    # weight clipping
    clip = options.clip
    # temp
    # tau = options.tau
    tau = [0.1, 0.2]  # manual override

    # encoding information
    size = options.decode_size
    n_values = 2 ** size
    n_categorical_variables = 1

    # initial gen model
    weights = None
    if options.init is not None:
        weights = options.init
        g_model = None
        # g_model = load_model(g_model_name, custom_objects={"GumbelSoftmaxActivation":GumbelSoftmaxActivation(n_categories=5  )})
    print('_' * 90)
    print('OPTIONS')
    print('_' * 90)
    print(' GAN model: {:<50}'.format(options.gan_model))
    print(' Training data file: {:<50}'.format(options.input))
    print(' Training mode: {:<50}'.format(
        'False' if options.train is False else 'True'))
    if options.train is not False:
        print('     Output model name: {:<50}'.format(options.output))
        print('     Training epochs: {:<50}'.format(str(options.EPOCHS)))
        print('     input_dim of latent space: {:<50}'.format(str(options.latent)))
        print('     Batch size: {:<50}'.format(str(options.batch)))
        print('     k hyperparameter: {:<50}'.format(str(options.k_hyp)))
        print('     Unrolling hyperparameter: {:<50}'.format(str(options.unrolling)))
        print('     Learning rate: {:<50}'.format(str(options.lr)))
        print('     Weigth clipping: {:<50}'.format(str(options.clip)))
        print('     Decoding size: {:<50}'.format(str(size)))
        print('     Gumbel-Softmax temperature: {:<50}'.format(str(tau)))
    if options.init is not None:
        print(' Initial generator model: {:<50}'.format(options.init))
        # g_model.summary()
    print('_' * 90)
    # _________________________________________________________________________________________

    # GPU memory usage configuration
    cfg = tf.compat.v1.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=cfg)
    tf.compat.v1.keras.backend.set_session(sess)
    # _________________________________________________________________________________________

    # load and scale data
    print('Loading data: '+inputfile)
    dataset = load(inputfile)
    dataset, encoder = scale(dataset)
    input_dim = dataset[1].shape[1]
    print('Data succesfully loaded')
    # _________________________________________________________________________________________

    # Create and train model
    GAN = getattr(importlib.import_module('models.' + gan_m), gan_m)
    model = GAN(input_dim=input_dim,
                latent_dim=latent_dim,
                n_categorical_variables=n_categorical_variables,
                learning_rate=learning_rate,
                generator=None,
                tau=tau)
    # create the models
    model.define_gan()
    if weights is not None:
        model.generator.load_weights(weights)
    model.generator.summary()
    model.critic.summary()
    model.gan.summary()
    # train the model
    if options.train is not False:
        _ = train(model, dataset, latent_dim, n_discrim_updates,
                  n_gen_updates, epochs, nbatch, n_values)
    # _________________________________________________________________________________________

    # Evaluate the model
    if model.generator is not None:
        activations, activations_gen = generate_evaluation_samples(model.generator, dataset, latent_dim)
        activations = encoder.inverse_transform(activations)
        activations_gen = encoder.inverse_transform(activations_gen)

        evaluation = Evaluation(activations[:, 0], activations_gen[:, 0], encoder.categories_[0].size)
        evaluation.calculate_parameters()
        evaluation.print_results()
        evaluation.plot_hits()
        evaluation = Evaluation(activations[:, 1], activations_gen[:, 1], encoder.categories_[1].size)
        evaluation.calculate_parameters()
        evaluation.print_results()
        evaluation.plot_hits()

    else:
        fake = None
        print("=" * 90)
        print('Warning: There is no generator model to evaluate')
        print("=" * 90)
    # _________________________________________________________________________________________

    # Generate a few
    # generate_and_save(g_model, dataset, latent_dim, 300000, encoder, output)
    # _________________________________________________________________________________________
