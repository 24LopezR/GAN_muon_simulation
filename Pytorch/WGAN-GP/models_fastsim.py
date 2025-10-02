import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Generator(nn.Module):
    def __init__(self, n_output_variables, n_input_variables, n_radius, latent_dim):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.n_output = n_output_variables
        self.n_input = n_input_variables
        self.n_radius = n_radius
        
        self.generator_model = nn.Sequential(
            nn.Linear(in_features  = latent_dim + n_input_variables + n_radius, 
                      out_features = 32),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Linear(in_features  = 32, 
                      out_features = 64),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Linear(in_features  = 64, 
                      out_features = 128),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Linear(in_features  = 128, 
                      out_features = 4),
        )

    def forward(self, input_data):
        return self.generator_model(input_data)

    def sample_latent(self, num_samples):
        return torch.randn((num_samples, self.latent_dim), dtype=torch.float)


class Discriminator(nn.Module):
    def __init__(self, n_output_variables, n_input_variables, n_radius):
        super(Discriminator, self).__init__()

        self.n_output = n_output_variables
        self.n_input = n_input_variables
        self.n_radius = n_radius
        
        self.critic_model = nn.Sequential(
            nn.Linear(in_features  = n_output_variables + n_input_variables + n_radius, 
                      out_features = 128),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Linear(in_features  = 128, 
                      out_features = 64),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Linear(in_features  = 64, 
                      out_features = 32),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.Linear(in_features  = 32, 
                      out_features = 1),
        )

    def forward(self, input_data):
        return self.critic_model(input_data)
