import torch
import torch.nn as nn
import torch.nn.functional as F
from ..standalone_layers import LightWeightGatedConv2D

from typing import List, Dict, Any

class BaseTemporalFeatExtractor(nn.Module):
    def __init__(self):
        super(BaseTemporalFeatExtractor, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'BaseTemporalFeatExtractor':
        if config['name'] == 'TemporalFeatureExtractor' and config['version'] == 1.0:
            return TemporalFeatExtractor.from_config(config)
        else:
            assert False, f"Unknown config: {config}"

class TemporalFeatExtractor(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int] = [32, 32]):
        super(TemporalFeatExtractor, self).__init__()
        netlist = []
        c1 = in_channels
        for i in range(len(layers)):
            netlist.append(LightWeightGatedConv2D(c1, layers[i], kernel_size=3, padding=1, stride=1))
            netlist.append(nn.ReLU(inplace=True))
            c1 = layers[i]
        netlist.append(LightWeightGatedConv2D(c1, out_channels, kernel_size=3, padding=1, stride=1))
        netlist.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*netlist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'TemporalFeatExtractor':
        return TemporalFeatExtractor(config['in_channels'], config['out_channels'], config['layers'])
    