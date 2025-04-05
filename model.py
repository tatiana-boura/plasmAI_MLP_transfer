import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, h1, num_layers, freeze_layers, output_size):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()  # List to hold layers

        # First layer
        self.layers.append(nn.Linear(2, h1))

        # Add hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(h1, h1))  # Each hidden layer has h1 neurons

        # Output layer
        self.out = nn.Linear(h1, output_size)

        if len(freeze_layers) > 0:
            print('Freezing layers.')

            # Freeze selected layers
            for layer_idx in freeze_layers:
                if layer_idx < len(self.layers):  # Ensure valid layer index
                    for param in self.layers[layer_idx].parameters():
                        param.requires_grad = False

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.elu(self.layers[i](x))  # Apply ELU after each layer
        x = self.out(x)  # Output layer
        return x
