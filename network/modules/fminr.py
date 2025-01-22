# Fourier mapped implicit neural representation
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.utils import make_coord
from utils.image_utils import ImageUtils

from typing import List

class FourierMappedINR(nn.Module):
    def __init__(self, lr_feat_c: int = 32, gb_feat_c: int = 32, out_channels: int = 12, mlp_inp_channels: int = 64, hidden_channels: List[int] = [64, 64, 64]):
        """Fourier mapped implicit neural representation module.

        Args:
            lr_feat_c (int): Number of channels in the low-resolution feature map.
            gb_feat_c (int): Number of channels in the G-buffer feature map.
            out_channels (int): Number of output channels.
            hidden_channels (List[int]): List of hidden channels for the neural network.
        """
        super(FourierMappedINR, self).__init__()

        self.lr_feat_c = lr_feat_c
        self.gb_feat_c = gb_feat_c
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.mlp_inp_channels = mlp_inp_channels

        # Define the neural network
        self.amplitude_conv = nn.Conv2d(lr_feat_c, mlp_inp_channels, kernel_size=3, padding=1)
        self.freq_conv = nn.Conv2d(gb_feat_c, mlp_inp_channels, kernel_size=3, padding=1)
        self.phase_conv = nn.Conv2d(1, mlp_inp_channels//2, kernel_size=1, bias=False)

        self.mlp = self._make_siren_mlp(omega_0=30.0)
    
    def forward(self, lr_feat: torch.Tensor, gb_feat: torch.Tensor, upscale_factor: float = 2.0) -> torch.Tensor:
        """Forward pass of the Fourier mapped INR module.

        Args:
            lr_feat (torch.Tensor): Low-resolution feature map. In format (B, C, H, W).
            gb_feat (torch.Tensor): G-buffer feature map. In format (B, C, H, W).
            upscale_factor (float): Upscale factor.
        """
        
        n, _, lr_h, lr_w = lr_feat.shape
        _, _, gb_h, gb_w = gb_feat.shape
        device = lr_feat.device

        # Create a tensor of values upscale_factor of dimension (n, 1, gb_h, gb_w)
        upscale_factor_tensor = torch.full((n, 1, gb_h, gb_w), upscale_factor, device=device, dtype=torch.float32)
        upscale_factor_inv_tensor = 1.0 / upscale_factor_tensor

        # Coordinate grids
        coord_gb = make_coord((gb_h, gb_w), device=device).unsqueeze(0).repeat(n, 1, 1)
        # No idea how it happens, Black fucking magic
        # All I know is it computes the relative coordinates(distance) between hr_pixels and their nearest lr_pixels
        lr_feat_coord = make_coord((lr_h, lr_w), device=device, flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(n, 2, lr_h, lr_w)
        q_coord = F.grid_sample(
            lr_feat_coord, coord_gb.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)
        rel_coord = coord_gb - q_coord
        rel_coord[:, :, 0] *= lr_h
        rel_coord[:, :, 1] *= lr_w
        my_rel_coord = rel_coord.permute(0, 2, 1).view(n, 2, gb_h, gb_w)

        # Compute the amplitude, frequency, and phase components
        lr_amplitude = self.amplitude_conv.forward(lr_feat)
        hr_freq = self.freq_conv.forward(gb_feat)
        hr_phase = self.phase_conv.forward(upscale_factor_inv_tensor)
        # Upsample the amplitude using the nearest pixel position
        upsampled_amplitude = F.grid_sample(
            lr_amplitude,
            coord_gb.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False
        )[:, :, 0, :].view(n, -1, gb_h, gb_w).contiguous()
        upsampled_amplitude = torch.cat([upsampled_amplitude], dim=1)
        
        hr_freq = torch.stack(torch.split(hr_freq, 2, dim=1), dim=1)
        hr_freq = torch.mul(hr_freq, my_rel_coord.unsqueeze(1)).sum(2)
        hr_freq += hr_phase
        hr_freq = torch.cat((torch.cos(np.pi * hr_freq), torch.sin(np.pi * hr_freq)), dim=1)
        mlp_inp = upsampled_amplitude * hr_freq
        
        # Compute the output
        out = self.mlp(mlp_inp)
        return out


    def _make_mlp(self) -> nn.Sequential:
        """Create the MLP network.
        """

        layers = []
        inp_channels = self.mlp_inp_channels
        for hidden_channel in self.hidden_channels:
            layers.append(nn.Conv2d(inp_channels, hidden_channel, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            inp_channels = hidden_channel
        layers.append(nn.Conv2d(inp_channels, self.out_channels, kernel_size=1))
        return nn.Sequential(*layers)
    
    def _make_siren_mlp(self , omega_0) -> nn.Sequential:
        """Create the MLP network with SIREN activation."""
        layers = []
        inp_channels = self.mlp_inp_channels
        for hidden_channel in self.hidden_channels:
            layers.append(nn.Conv2d(inp_channels, hidden_channel, kernel_size=1))
            layers.append(SineActivation(omega_0=omega_0))  # Use SIREN activation with omega_0 scaling
            inp_channels = hidden_channel
        layers.append(nn.Conv2d(inp_channels, self.out_channels, kernel_size=1))  # No activation on output layer
        return nn.Sequential(*layers)

class SineActivation(nn.Module):
    """Sine activation layer for SIREN with omega_0 scaling."""
    def __init__(self, omega_0: float = 30.0):
        super(SineActivation, self).__init__()
        self.omega_0 = omega_0  # Store omega_0 as an attribute

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply omega_0 scaling to the sine activation
        return torch.sin(self.omega_0 * x)