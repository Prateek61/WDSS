# Spatial and Temporal masks

from utils.image_utils import upsample
import numpy as np
import torch
from torch import Tensor

def get_spatial_mask(hr_depth: Tensor, hr_normal: Tensor, hr_albedo: Tensor, lr_depth: Tensor, lr_normal: Tensor, lr_albedo: Tensor, depth_const: float, normal_const: float, albedo_const: float) -> Tensor:

    upsample_factor = hr_depth.shape[-1] / lr_depth.shape[-1]
    upsampled_depth = upsample(lr_depth, upsample_factor)
    upsampled_normal = upsample(lr_normal, upsample_factor)
    upsampled_albedo = upsample(lr_albedo, upsample_factor)

    depth_diff = torch.abs(hr_depth - upsampled_depth)
    # Subtract depth const from depth_diff
    depth_diff = depth_diff - depth_const
    # Apply Heaviside step function, gives 1 if depth_diff > 0 else 0
    depth_mask = torch.heaviside(depth_diff, 0)

    albedo_diff = torch.abs(hr_albedo - upsampled_albedo)
    # Subtract albedo const from albedo_diff
    albedo_diff = albedo_diff - albedo_const
    # Apply Heaviside step function, gives 1 if albedo_diff > 0 else 0
    albedo_mask = torch.heaviside(albedo_diff, 0)

    # Perform dot product between normal and upsampled_normal
    # This is the cosine similarity between the two normal vectors
    normal_dotted = torch.sum(hr_normal * upsampled_normal, dim=1)
    # Subtract normal_dotted from normal_const
    normal_diff = normal_const - normal_dotted
    # Apply Heaviside step function, gives 1 if normal_diff > 0 else 0
    normal_mask = torch.heaviside(normal_diff, 0)

    return depth_mask, normal_mask, albedo_mask

    
    