import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, n_neurons):
        super(Encoder, self).__init__()
        self.model = nn.Sequential()
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
    def __init__(self, output_dim, n_neurons):
        super(Decoder, self).__init__()
        self.model = nn.Sequential()
        for i in range(len(n_neurons)-1, -1, -1):
            if i == len(n_neurons)-1:
                n_in = n_neurons[i]
            else:
                n_in = n_neurons[i+1]
            n_out = n_neurons[i]
            self.model.add_module(f'dec{i + 1}', nn.Linear(n_in, n_out))
            self.model.add_module(f'dec{i + 1}_relu', nn.ReLU())

        self.model.add_module('output', nn.Linear(n_neurons[0], output_dim))

    def forward(self, x):
        return self.model(x)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoder_neurons, decoder_neurons):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, encoder_neurons)
        self.decoder = Decoder(input_dim, decoder_neurons)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded