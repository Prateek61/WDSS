import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class LRFrameFeatureExtractor(nn.Module):
    def __init__(self, in_channels:int, out_channels: int, layers: List[int] = [32,32,32]):
        """Low resolution frame feature extractor.
            using conv layers to extract features from low resolution frames.
        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            layers (List[int]): The number of feature maps in each layer.
        """
        super(LRFrameFeatureExtractor, self).__init__()
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

       