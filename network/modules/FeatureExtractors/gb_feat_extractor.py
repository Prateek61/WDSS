import torch
import torch.nn as nn
import torch.nn.functional as F
from ..standalone_layers import ResBlock , doubleResidualConv

from typing import List, Dict, Any

class BaseGBFeatExtractor(nn.Module):
    def __init__(self):
        super(BaseGBFeatExtractor, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'BaseGBFeatExtractor':
        if config['name'] == 'GBFeatureExtractor' and config['version'] == 1.0:
            return GBFeatureExtractor.from_config(config)
        else:
            assert False, f"Unknown config: {config}"

class GBFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, num_layers: int = 5, layer_size: int = 64 , useDoubleResidualConv = True):
        super(GBFeatureExtractor, self).__init__()
        netlist = []

        if useDoubleResidualConv:
            for _ in range(num_layers - 1):
                netlist.append(nn.Conv2d(in_channels, layer_size, kernel_size=3, padding=1, stride=1))
                netlist.append(nn.ReLU())
                netlist.append(doubleResidualConv(layer_size))
                in_channels = layer_size

        else:
            netlist.append(nn.Conv2d(in_channels, layer_size, kernel_size=3, padding=1, stride=1))
            netlist.append(nn.ReLU())

            for i in range(num_layers - 1):
                netlist.append(ResBlock(layer_size))
                netlist.append(nn.ReLU())

        self.net = nn.Sequential(*netlist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'GBFeatureExtractor':
        return GBFeatureExtractor(config['in_channels'], config['num_layers'], config['layer_size'] , False) 