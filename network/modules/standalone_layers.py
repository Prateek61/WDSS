import torch
import torch.nn as nn
import torch.nn.functional as F
from network.commons import pad_tensor


class ResBlock(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.
    Diagram:
        ---Conv1x1--Conv3x3-+-Conv1x1--+--
                   |________|
         |_____________________________|
    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats: int, ratio: int = 2):
        super(ResBlock, self).__init__()

        self.expand_conv = nn.Conv2d(n_feats, int(ratio*n_feats), 1, 1, 0)
        self.fea_conv = nn.Conv2d(int(ratio*n_feats), int(ratio*n_feats), 3, 1, 1)
        self.reduce_conv = nn.Conv2d(int(ratio*n_feats), n_feats, 1, 1, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.expand_conv(x)
        out_identify: torch.Tensor = out

        # explicitly padding with bias for reparameterizing in the test phase
        b0: torch.Tensor = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identify
        out = self.reduce_conv(out)
        out += x

        return out
    

class LightWeightGatedConv2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(LightWeightGatedConv2D, self).__init__()

        self.feature = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x) * self.gate(x)