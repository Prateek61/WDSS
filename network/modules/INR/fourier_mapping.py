import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import make_coord
import utils.image_utils as ImageUtils

from typing import List, Dict, Any

class FourierMapping(nn.Module):
    def __init__(
           self,
           lr_feat_c: int,
           gb_feat_c: int,
           mapped_c: int
    ):
        super(FourierMapping, self).__init__()

        self.amplitude_conv = nn.Conv2d(lr_feat_c, mapped_c, kernel_size=3, padding=1)
        self.freq_conv = nn.Conv2d(gb_feat_c, mapped_c, kernel_size=3, padding=1)
        self.phase_conv = nn.Conv2d(1, mapped_c//2, kernel_size=3, padding=1)

    def forward(self, lr_feat: torch.Tensor, gb_feat: torch.Tensor, upscale_factor: float) -> torch.Tensor:
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
        mapped_out = upsampled_amplitude * hr_freq

        return mapped_out

    @staticmethod 
    def from_config(config: Dict[str, Any]) -> 'FourierMapping':
        return FourierMapping(config['lr_feat_c'], config['gb_feat_c'], config['mapped_c'])
    
class SimpleMapping(nn.Module):
    def __init__(
        self,
        lr_feat_c: int,
        gb_feat_c: int,
        mapped_c: int
    ):
        super(SimpleMapping, self).__init__()

        self.mapping = nn.Conv2d(lr_feat_c + gb_feat_c, mapped_c, kernel_size=3, padding=1)

    def forward(self, lr_feat: torch.Tensor, gb_feat: torch.Tensor, upscale_factor: float) -> torch.Tensor:
        # Upsample the lr_feat
        lr_feat_upsampled = ImageUtils.upsample(lr_feat, upscale_factor)
        return self.mapping(torch.cat([lr_feat_upsampled, gb_feat], dim=1))
