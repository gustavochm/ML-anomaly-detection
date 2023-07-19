"""
Implementation of an autoencoder

input_dim : int
  number of features of the dataset to be input to the first layer of the
  autoencoder.
n_neurons: list
  Sequence of integers that represent the number of neurons of each layer.
latent_dim: int
  Size of the latent dimension, which is the output of the encoder and the
  input of the decoder submodels.
"""

import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_dim, n_neurons):
        super(Encoder, self).__init__()
        self.model = nn.Sequential()
        # Loop through layers
        for i in range(len(n_neurons)):
            if i == 0:
                n_in = input_dim
            else:
                n_in = n_neurons[i - 1]
            n_out = n_neurons[i]
            self.model.add_module(f'enc{i + 1}', nn.Linear(n_in, n_out))
            self.model.add_module(f'enc{i + 1}_relu', nn.ReLU())

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):

    def __init__(self, latent_dim, n_neurons):
        super(Decoder, self).__init__()
        self.model = nn.Sequential()
        # Loop through layers
        for i in range(len(n_neurons)):
            if i == 0:
                n_in = latent_dim
            else:
                n_in = n_neurons[i - 1]
            n_out = n_neurons[i]
            self.model.add_module(f'dec{i + 1}', nn.Linear(n_in, n_out))
            self.model.add_module(f'dec{i + 1}_relu', nn.ReLU())

    def forward(self, z):
        return self.model(z)


class AEModel(nn.Module):
    """ Implement an autoencoder in PyTorch. """
    def __init__(self, input_dim, n_neurons):
        # input_dim = 50 for TE detection without windows
        super(AEModel, self).__init__()
        # Separate n_neuron list in two parts
        n_half_layers = len(n_neurons) // 2
        encoder_neurons = n_neurons[:n_half_layers]
        decoder_neurons = n_neurons[n_half_layers:]
        # Build encoder-decoder system
        self.encoder = Encoder(input_dim=input_dim,
                               n_neurons=encoder_neurons)
        self.decoder = Decoder(latent_dim=encoder_neurons[-1],
                               n_neurons=decoder_neurons)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
