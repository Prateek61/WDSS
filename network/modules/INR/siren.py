import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, trainable=False):
        super().__init__()

        if trainable:
            self.omega_0 = nn.Parameter(torch.tensor([omega_0], dtype=torch.float32))
        else:
            self.omega_0 = omega_0

        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                if isinstance(self.omega_0, nn.Parameter):
                    # Using magic number 30 as in the paper
                    omega_0 = self.omega_0.data.item()
                else:
                    omega_0 = self.omega_0
                    
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / omega_0, 
                                             np.sqrt(6 / self.in_features) / omega_0,)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30, trainable=False):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, trainable=trainable))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, trainable=trainable))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = np.sqrt(6 / hidden_features) / hidden_omega_0
                nn.init.uniform_(final_linear.weight, -bound, bound)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0, trainable=trainable))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        """ Process spatial inputs (batch_size, in_features, height, width). """
        batch_size, channels, height, width = coords.shape

        # Reshape to (batch_size, channels, H*W) and transpose to (batch_size, H*W, channels)
        coords = coords.view(batch_size, channels, -1).permute(0, 2, 1)  # (B, H*W, C)

        # Process each spatial location independently
        output = self.net(coords)  # (B, H*W, out_features)

        # Reshape back to (batch_size, out_features, height, width)
        output = output.permute(0, 2, 1).view(batch_size, -1, height, width)

        return output

    @staticmethod
    def from_config(config):
        return Siren(
            config['in_features'], config['hidden_features'], config['hidden_layers'],
            config['out_features'], config['outermost_linear'], config['first_omega_0'],
            config['hidden_omega_0'], config['trainable']
        )


if __name__ == '__main__':
    # Test with spatial input (1, 64, 32, 32)
    model = Siren(64, 64, 3, 3, outermost_linear=True, first_omega_0=30, hidden_omega_0=30, trainable=True)
    x = torch.randn(1, 64, 32, 32)  # (batch_size, channels, height, width)
    output = model(x)
    print(output.shape)  # Expected: (1, 3, 32, 32)
