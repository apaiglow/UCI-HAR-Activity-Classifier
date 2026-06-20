import torch.nn as nn

class MyNN(nn.Module):

    def __init__(self, input_dim, output_dim, num_hidden_layer, neuron_per_layer, dropout_ratio):

        super().__init__()

        layers = []

        for i in range(num_hidden_layer):

            layers.append(nn.Linear(input_dim, neuron_per_layer))
            layers.append(nn.BatchNorm1d(neuron_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_ratio))
            input_dim = neuron_per_layer
        
        layers.append(nn.Linear(neuron_per_layer, output_dim))

        #*layers unpacks a list as Sequential requires unpacked list
        self.model = nn.Sequential(*layers)

    def forward(self, x):

        return self.model(x)