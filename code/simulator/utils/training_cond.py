##################################################################################################
#### Training module                                                                          ####
##################################################################################################
from numpy import ones
from matplotlib import pyplot
from keras import backend

from .generating_cond import generate_real_samples, generate_fake_samples, generate_latent_points

def plot_history(d1_hist, d2_hist, g_hist):
	"""
	Plots the discriminator loss for real and fake samples and the generator loss, and saves the plot to a .png file.
	Arguments:
	Input:
		d1_hist: discriminator loss on real samples
		d2_hist: discriminator loss on fake samples
		g_hist: generator loss
	"""
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png', dpi = 400)
	pyplot.close()

def save_model(epoch, g_model):
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, lr_decay, n_discrim_updates, n_gen_updates, n_epochs=10, n_batch=128, weights=None):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # lists for keeping track of loss (lari)
    # manually enumerate epochs
    d_real_hist, d_fake_hist, g_hist = list(), list(), list()
    for i in range(n_epochs):
        # set learning rate for epoch
        learning_rate = lr_decay(i+1)
        backend.set_value(d_model.optimizer.learning_rate, learning_rate)
        backend.set_value(gan_model.optimizer.learning_rate, learning_rate)
        #print('> Epoch: %d -- learning_rate = %.5f' % (i+1, scifor(learning_rate)))
        # enumerate batches over the training set
        for j in range(bat_per_epo):
            for k in range(n_discrim_updates):
                # get randomly selected 'real' samples
                [X_real, labels_real], w_real, y_real = generate_real_samples(dataset, half_batch, weights=weights, ind=i)
                # update discriminator model weights
                d_loss1, acc_real = d_model.train_on_batch([X_real, labels_real], y_real, sample_weight=w_real)
				# generate 'fake' examples
                [X_fake, labels], w_fake, y_fake = generate_fake_samples(g_model, dataset, latent_dim, half_batch, weights=weights, ind=i)
				# update discriminator model weights
                d_loss2, acc_fake = d_model.train_on_batch([X_fake, labels], y_fake, sample_weight=w_fake)
            for k in range(n_gen_updates):
       			# prepare points in latent space as input for the generator
                [z_input, labels_input], w_input = generate_latent_points(dataset, latent_dim, n_batch, weights=weights, ind=i)
    			# create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
    			# update the generator via the discriminator's error
                g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan, sample_weight=w_input)
            if (j+1) % 20 == 0:
                print('>{:<3}, {:<4}/{:<4}, d1={:<8.3}, d2={:<8.3}, g={:<8.3}, acc_real={:<4.3%}, acc_fake={:<4.3%}'.format(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss, acc_real, acc_fake))
                d_real_hist.append(d_loss1)
                d_fake_hist.append(d_loss2)
                g_hist.append(g_loss)
        plot_history(d_real_hist,d_fake_hist,g_hist)
        # evaluate the model performance, sometimes
        if (i+1) % 5 == 0:
            save_model(i, g_model)
    # save the generator model
    g_model.save('gan_generator.h5')
    return g_model