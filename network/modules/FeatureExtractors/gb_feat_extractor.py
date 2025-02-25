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
        elif config['name'] == 'GBFeatureExtractor' and config['version'] == 2.0:
            return GBFeatureExtractorDoubleResidual.from_config(config)
        else:
            assert False, f"Unknown config: {config}"

class GBFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, num_layers: int = 5, layer_size: int = 64 , useDoubleResidualConv = False):
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
    
class GBFeatureExtractorDoubleResidual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_layers: int = 3, layer_size: int = 64):
        super(GBFeatureExtractorDoubleResidual, self).__init__()
        netlist = []

        if num_layers < 2:
            raise ValueError("Number of layers must be at least 2")

        netlist.append(nn.Conv2d(in_channels, layer_size, kernel_size=3, padding=1, stride=1))
        netlist.append(nn.ReLU())
        netlist.append(doubleResidualConv(layer_size))

        for i in range(num_layers - 2):
            netlist.append(nn.Conv2d(layer_size, layer_size, kernel_size=3, padding=2, stride=1, dilation=2))
            netlist.append(nn.ReLU())
            netlist.append(doubleResidualConv(layer_size))

        netlist.append(nn.Conv2d(layer_size, out_channels, kernel_size=3, padding=2, stride=1, dilation=2))
        netlist.append(nn.ReLU())
        netlist.append(doubleResidualConv(out_channels))

        self.net = nn.Sequential(*netlist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'GBFeatureExtractorDoubleResidual':
        return GBFeatureExtractorDoubleResidual(config['in_channels'], config['out_channels'], config['num_layers'], config['layer_size'])