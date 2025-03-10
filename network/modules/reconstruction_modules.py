import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from typing import List, Dict, Any

class BaseFeatureFusion(nn.Module):
    def __init__(self):
        super(BaseFeatureFusion, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'BaseFeatureFusion':
        if config['name'] == 'FeatureFusion' and config['version'] == 1.0:
            return FeatureFusion.from_config(config)
        else:
            assert False, f"Unknown config: {config}"

class FeatureFusion(BaseFeatureFusion):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int] = [64, 48]):
        super(FeatureFusion, self).__init__()
        netlist = []
        c1 = in_channels
        for i in range(len(layers)):
            netlist.append(nn.Conv2d(c1, layers[i], kernel_size=3, padding=1, stride=1))
            netlist.append(nn.ReLU())
            c1 = layers[i]
        netlist.append(nn.Conv2d(c1, out_channels, kernel_size=3, padding=1, stride=1))
        self.net = nn.Sequential(*netlist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'FeatureFusion':
        return FeatureFusion(config['in_channels'], config['out_channels'], config['layers'])
    
