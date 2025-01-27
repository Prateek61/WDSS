import torch    
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import List

class ComplexGaborConv2d(nn.Module):
    """2D Convolutional layer with complex Gabor nonLinearity
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = True, is_first: bool = False, is_last: bool = False, omega0: float = 10.0, sigma0: float = 10.0, trainable: bool = False):
        super(ComplexGaborConv2d, self).__init__()


        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_last = is_last

        if is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        self.omega_0 = nn.Parameter(omega0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(sigma0*torch.ones(1), trainable)

        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=bias,
                              dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        omega = self.omega_0 * x
        scale = self.scale_0 * x
        out = torch.exp(1j * omega - scale.abs().square())
        if self.is_last:
            out = out.real
        return out
    
class WIREINR_Conv2D(nn.Module):
    """WireINR Conv2D module
    """
    def __init__(self, in_channels: int, out_channels: int, omega: float = 10.0, sigma: float = 10.0, trainable: bool = False, hidden_layers: List[int] = [64, 64, 64, 64]):
        super(WIREINR_Conv2D, self).__init__()

        # Since complex numbers are two real numbers
        # reduce the number of hidden parameters by 2
        hidden_channels = [int(hidden_channel/np.sqrt(2)) for hidden_channel in hidden_layers]

        net: List[nn.Module] = []
        net.append(ComplexGaborConv2d(in_channels, hidden_channels[0], kernel_size=1, omega0=omega, sigma0=sigma, is_first=True, trainable=trainable))
        for i in range(1, len(hidden_channels)):
            net.append(ComplexGaborConv2d(hidden_channels[i-1], hidden_channels[i], kernel_size=1, omega0=omega, sigma0=sigma, trainable=trainable))
        net.append(nn.Conv2d(hidden_channels[-1], out_channels, kernel_size=1, dtype=torch.cfloat))

        self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).real
    