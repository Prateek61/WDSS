import torch
import torch.nn as nn

from typing import List, Dict, Any

class ReLUINR(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int = 4,
            layer_size: int = 64
    ):
        super(ReLUINR, self).__init__()
        netlist = []

        netlist.append(nn.Conv2d(in_channels, layer_size, kernel_size=1, padding=0, stride=1))
        netlist.append(nn.ReLU())

        for i in range(num_layers - 2):
            netlist.append(nn.Conv2d(layer_size, layer_size, kernel_size=1, padding=0, stride=1))
            netlist.append(nn.ReLU())

        netlist.append(nn.Conv2d(layer_size, out_channels, kernel_size=1, padding=0, stride=1))
        netlist.append(nn.ReLU())

        self.net = nn.Sequential(*netlist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def from_config(config: Dict[str, Any]) -> 'ReLUINR':
        return ReLUINR(config['in_channels'], config['out_channels'], config['num_layers'], config['layer_size'])
    