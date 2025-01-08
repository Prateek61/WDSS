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