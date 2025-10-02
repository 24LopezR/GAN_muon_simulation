import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from models_fastsim import Generator, Discriminator
from training import Trainer
from MuonDataset import MuonDataset

DATAFILESPATH = '/home/ruben/Documents/Samples_csv/'
BATCH_SIZE    = 4096
N_IN_VARS     = 4
N_OUT_VARS    = 4
N_RADIUS      = 10
LATENT_DIM    = 16

dataset = MuonDataset(DATAFILESPATH)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

## Data loading test...
in_vars, out_vars, radius = next(iter(dataloader))
print(f"In batch shape:  {in_vars.size()}")
print(f"Out batch shape: {out_vars.size()}")
print(f"Radius shape:    {radius.size()}")

## Define the models
generator     = Generator(n_output_variables=N_OUT_VARS,
                          n_input_variables=N_IN_VARS,
                          n_radius=N_RADIUS,
                          latent_dim=LATENT_DIM)
critic        = Discriminator(n_output_variables=N_OUT_VARS,
                              n_input_variables=N_IN_VARS, 
                              n_radius=N_RADIUS)

print(generator)
print(critic)

# Initialize optimizers
lr = 1e-4
betas = (.9, .99)
G_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=betas)
D_optimizer = optim.Adam(critic.parameters(),    lr=lr, betas=betas)

# Train model
epochs = 50
print('>> CUDA: {0}'.format(torch.cuda.is_available()))
trainer = Trainer(generator, critic, G_optimizer, D_optimizer,
                  use_cuda=torch.cuda.is_available())
trainer.train(dataloader, epochs)

# Save models
name = ''
torch.save(trainer.G.state_dict(), './gen_' + name + '.pt')
torch.save(trainer.D.state_dict(), './dis_' + name + '.pt')
