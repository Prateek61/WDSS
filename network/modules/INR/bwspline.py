import torch    
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Any

class BWSpline(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int = 4,
            layer_size: int = 64,
            scale: float = 1.0
    ):
        super(BWSpline, self).__init__()
        netlist = []

        netlist.append(nn.Conv2d(in_channels, layer_size, kernel_size=1, padding=0, stride=1))
        netlist.append(BSplineWavelet(scale=scale))

        for i in range(num_layers - 2):
            netlist.append(nn.Conv2d(layer_size, layer_size, kernel_size=1, padding=0, stride=1))
            netlist.append(BSplineWavelet(scale=scale))

        netlist.append(nn.Conv2d(layer_size, out_channels, kernel_size=1, padding=0, stride=1))
        netlist.append(BSplineWavelet(scale=scale))

        self.net = nn.Sequential(*netlist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def from_config(config: Dict[str, Any]) -> 'BWSpline':
        return BWSpline(config['in_channels'], config['out_channels'], config['num_layers'], config['layer_size'], config['scale'])
    


class BSplineWavelet(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = torch.as_tensor(scale)
    
    def forward(self, x):
        output = bspline_wavelet(x, self.scale)
        
        return output
    

def bspline_wavelet(x, scale):
    return (1 / 6) * F.relu(scale*x)\
    - (8 / 6) * F.relu(scale*x - (1 / 2))\
    + (23 / 6) * F.relu(scale*x - (1))\
    - (16 / 3) * F.relu(scale*x - (3 / 2))\
    + (23 / 6) * F.relu(scale*x - (2))\
    - (8 / 6) * F.relu(scale*x - (5 / 2))\
    +(1 / 6) * F.relu(scale*x - (3))