import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_tensor(t: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    """Returns a padded tensor with the given pattern.

    Args:
        t (torch.Tensor): Input tensor.
        pattern (torch.Tensor): Pattern tensor.

    Returns:
        torch.Tensor: Padded tensor.
    """
    
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t

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
        self.reparamed = False
        self.n_feats = n_feats
        self.ratio = ratio
        self.reparamed_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if self.reparamed:
        #     return self.reparamed_conv(x)

        out: torch.Tensor = self.expand_conv(x)
        out_identify: torch.Tensor = out

        # explicitly padding with bias for reparameterizing in the test phase
        # b0: torch.Tensor = self.expand_conv.bias
        # out = pad_tensor(out, b0)

        out = self.fea_conv(out) + out_identify
        out = self.reduce_conv(out)
        out += x

        return out
    
    def reparameterize(self):
        # Merge conv1x1, conv3x3, and conv1x1 into a single conv layer
        if self.reparamed:
            return
        
        device = self.expand_conv.weight.device
        
        # Get the weights of the convolution layers
        w1 = self.expand_conv.weight
        w2 = self.fea_conv.weight
        w3 = self.reduce_conv.weight
        b1 = self.expand_conv.bias
        b2 = self.fea_conv.bias
        b3 = self.reduce_conv.bias
        
        mid_feats, n_feats = w1.shape[:2]

        # First step: remove the middle identity
        w2_mod = w2.detach().clone()
        for i in range(mid_feats):
            w2_mod[i, i, 1, 1] += 1.0

        # Second step: fuse the first conv1x1 and conv3x3
        merged_k1k2 = F.conv2d(input=w2_mod, weight=w1.permute(1, 0, 2, 3))
        merged_b1b2 = b1.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3, device=device)
        merged_b1b2 = F.conv2d(input=merged_b1b2, weight=w2_mod, bias=b2)

        # Third step: merge the remaining 1x1 convolution
        merged_w1w2w3 = F.conv2d(input=merged_k1k2.permute(1, 0, 2, 3), weight=w3).permute(1, 0, 2, 3)
        merged_b1b2b3 = F.conv2d(input=merged_b1b2, weight=w3, bias=b3).view(-1)

        # Last step: remove the global identity
        for i in range(n_feats):
            merged_w1w2w3[i, i, 1, 1] += 1.0

        # Create the reparameterized convolution layer
        self.reparamed_conv = nn.Conv2d(
            in_channels=self.n_feats,
            out_channels=self.n_feats,
            kernel_size=3,
            padding=1,
            stride=1
        )
        self.reparamed_conv.weight = nn.Parameter(merged_w1w2w3)
        self.reparamed_conv.bias = nn.Parameter(merged_b1b2b3)

        def reparamed_forward(x: torch.Tensor) -> torch.Tensor:
            return self.reparamed_conv(x)
        
        self.forward = reparamed_forward
        self.reparamed = True

        del self.expand_conv
        del self.fea_conv
        del self.reduce_conv
    

class LightWeightGatedConv2D(nn.Module):
    """
    Lightweight Gated Convolutions (LWGC) module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for the convolution.
        stride (int): Stride for the convolution.
        padding (int): Padding for the convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super(LightWeightGatedConv2D, self).__init__()

        self.feature = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x) * self.gate(x)
    
    
class doubleResidualConv(nn.Module):
    def __init__(self, outc: int, kernel_size: int = 3, padding: int = 1):
        super(doubleResidualConv,self).__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(outc,outc,kernel_size=kernel_size,padding=padding),
            nn.ReLU(),
            nn.Conv2d(outc,outc,kernel_size=kernel_size,padding=padding),
            nn.ReLU()
        )
    def forward(self,x):
        return self.conv(x) + x