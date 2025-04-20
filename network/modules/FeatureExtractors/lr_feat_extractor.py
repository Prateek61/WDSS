import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Any

class BaseLRFeatExtractor(nn.Module):
    def __init__(self):
        super(BaseLRFeatExtractor, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
class LRFrameFeatureExtractor(BaseLRFeatExtractor):
    def __init__(self, in_channels: int, out_channels: int, layers: List[int] = [32, 48, 48]):
        super(LRFrameFeatureExtractor, self).__init__() 
        netlist = []
        c1 = in_channels
        for i in range(len(layers)):
            netlist.append(nn.Conv2d(c1, layers[i], kernel_size=3, padding=1, stride=1))
            netlist.append(nn.ReLU(inplace=True))
            c1 = layers[i]
        netlist.append(nn.Conv2d(c1, out_channels, kernel_size=3, padding=1, stride=1))
        netlist.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*netlist)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
