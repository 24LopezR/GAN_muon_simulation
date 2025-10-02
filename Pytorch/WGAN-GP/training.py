import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


class Trainer():
    def __init__(self, generator, discriminator, gen_optimizer, dis_optimizer,
                 gp_weight=10, critic_iterations=5, print_every=50,
                 use_cuda=False):
        self.G = generator
        self.G_opt = gen_optimizer
        self.D = discriminator
        self.D_opt = dis_optimizer
        self.losses = {'G': [], 'D': [], 'GP': [], 'gradient_norm': []}
        self.num_steps = 0
        self.use_cuda = use_cuda
        self.gp_weight = gp_weight
        self.critic_iterations = critic_iterations
        self.print_every = print_every

        if self.use_cuda:
            self.G.cuda()
            self.D.cuda()

    def _critic_train_iteration(self, data):
        (in_vars, out_vars, radius_label) = data
        # Get generated data
        batch_size = in_vars.size()[0]
        gen_out_vars = self.sample_generator(batch_size, in_vars, radius_label)
        if self.use_cuda:
            in_vars  = in_vars.cuda()
            out_vars = out_vars.cuda()
            gen_out_vars = gen_out_vars.cuda()
            radius_label = radius_label.cuda()

        # Calculate probabilities on real and generated data
        #in_vars      = Variable(in_vars)
        #out_vars     = Variable(out_vars)
        #radius_label = Variable(radius_label)
        real_data = torch.cat((out_vars, in_vars, radius_label), dim=1)
        gen_data  = torch.cat((gen_out_vars, in_vars, radius_label), dim=1)
        print(real_data.get_device())
        d_real      = self.D(real_data)
        d_generated = self.D(gen_data)

        # Get gradient penalty
        gradient_penalty = self._gradient_penalty(real_data, gen_data)
        self.losses['GP'].append(gradient_penalty.item())

        # Create total loss and optimize
        self.D_opt.zero_grad()
        d_loss = d_generated.mean() - d_real.mean() + gradient_penalty
        d_loss.backward()

        self.D_opt.step()

        # Record loss
        self.losses['D'].append(d_loss.item())

    def _generator_train_iteration(self, data):
        (in_vars, out_vars, radius_label) = data

        self.G_opt.zero_grad()

        # Get generated data
        batch_size = in_vars.size()[0]
        gen_out_vars = self.sample_generator(batch_size, in_vars, radius_label)
        if self.use_cuda:
            in_vars  = in_vars.cuda()
            out_vars = out_vars.cuda()
            gen_out_vars = gen_out_vars.cuda()
            radius_label = radius_label.cuda()
        
        # Calculate loss and optimize
        gen_data  = torch.cat((gen_out_vars, in_vars, radius_label), dim=1)
        d_generated = self.D(gen_data)
        g_loss = - d_generated.mean()
        g_loss.backward()
        self.G_opt.step()

        # Record loss
        self.losses['G'].append(g_loss.item())

    def _gradient_penalty(self, real_data, generated_data):
        batch_size = real_data.size()[0]

        # Calculate interpolation
        alpha = torch.rand(real_data.shape)
        if self.use_cuda:
            alpha = alpha.cuda()
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.use_cuda:
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.use_cuda else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(batch_size, -1)
        self.losses['gradient_norm'].append(gradients.norm(2, dim=1).mean().item())

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()

    def _train_epoch(self, data_loader):
        for i, data in enumerate(data_loader):
            #print(f'>> in:  {data[0].shape}')
            #print(f'>> out: {data[1].shape}')
            #print(f'>> rad: {data[2].shape}')
            self.num_steps += 1
            self._critic_train_iteration(data)
            # Only update generator every |critic_iterations| iterations
            if self.num_steps % self.critic_iterations == 0:
                self._generator_train_iteration(data)

            if i % self.print_every == 0:
                print("Iteration {}".format(i + 1))
                print("D: {}".format(self.losses['D'][-1]))
                print("GP: {}".format(self.losses['GP'][-1]))
                print("Gradient norm: {}".format(self.losses['gradient_norm'][-1]))
                if self.num_steps > self.critic_iterations:
                    print("G: {}".format(self.losses['G'][-1]))

    def train(self, data_loader, epochs, save_training_gif=True):
        for epoch in range(epochs):
            print("\nEpoch {}".format(epoch + 1))
            self._train_epoch(data_loader)

    def sample_generator(self, num_samples, in_vars, radius_label):
        latent_samples = Variable(self.G.sample_latent(num_samples))
        latent_data = torch.cat((latent_samples, in_vars, radius_label), dim=1)
        if self.use_cuda:
            latent_data = latent_data.cuda()
        generated_data = self.G(latent_data)
        return generated_data

    def sample(self, num_samples):
        generated_data = self.sample_generator(num_samples)
        # Remove color channel
        return generated_data.data.cpu().numpy()[:, 0, :, :]
