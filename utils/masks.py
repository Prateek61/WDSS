# Spatial and Temporal masks

import numpy as np
import torch
from torch import Tensor

from typing import Dict, Tuple

from .image_utils import ImageUtils

class Mask:
    @staticmethod
    def spatial_and_temporal_mask(
        hr_base_color: torch.Tensor,
        hr_normal: torch.Tensor,
        hr_depth: torch.Tensor,
        lr_base_color: torch.Tensor,
        lr_normal: torch.Tensor,
        lr_depth: torch.Tensor,
        motion_vector: torch.Tensor,
        temporal_hr_base_color: torch.Tensor,
        temporal_hr_normal: torch.Tensor,
        temporal_hr_depth: torch.Tensor,
        upscale_factor: float = 2.0,
        spatial_threasholds: Dict[str, float] = {
            'depth': 0.04,
            'normal': 0.4,
            'albedo': 0.1
        }
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the spatial and temporal masks.
        """

        # Upsample the low-resolution images
        upsampled_lr_base_color = ImageUtils.upsample(lr_base_color, upscale_factor)
        upsampled_lr_normal = ImageUtils.upsample(lr_normal, upscale_factor)
        upsampled_lr_depth = ImageUtils.upsample(lr_depth, upscale_factor)

        # Compute the spatial mask
        spatial_mask = Mask._spatial_mask(
            hr_depth, hr_base_color, hr_normal,
            upsampled_lr_depth, upsampled_lr_base_color, upsampled_lr_normal,
            spatial_threasholds, upscale_factor
        )

        # Warp the high-resolution images
        warped_hr_base_color = Mask.warp_frame(temporal_hr_base_color, motion_vector)
        warped_hr_normal = Mask.warp_frame(temporal_hr_normal, motion_vector)
        warped_hr_depth = Mask.warp_frame(temporal_hr_depth, motion_vector)

        # Compute the temporal mask
        temporal_mask = Mask._temporal_mask(
            upsampled_lr_base_color, warped_hr_base_color,
            upsampled_lr_depth, warped_hr_depth,
            upsampled_lr_normal, warped_hr_normal
        )

        return spatial_mask, temporal_mask

    @staticmethod
    def _spatial_mask(
        hr_depth: torch.Tensor,
        hr_base_color: torch.Tensor,
        hr_normal: torch.Tensor,
        upsampled_lr_depth: torch.Tensor,
        upsampled_lr_base_color: torch.Tensor,
        upsampled_lr_normal: torch.Tensor,
        threasholds: Dict[str, float] = {
            'depth': 0.04,
            'normal': 0.4,
            'albedo': 0.1
        },
        upscale_factor: float = 2.0
    ) -> torch.Tensor:
        # Compute the masks
        depth_mask = Mask._depth_mask(hr_depth, upsampled_lr_depth, threasholds['depth'])
        normal_mask = Mask._normal_mask(hr_normal, upsampled_lr_normal, threasholds['normal'])
        albedo_mask = Mask._albedo_mask(hr_base_color, upsampled_lr_base_color, threasholds['albedo'])

        # Combine the masks
        spatial_mask = depth_mask + normal_mask + albedo_mask

        # Clamp the mask to 1
        spatial_mask = torch.clamp(spatial_mask, min=0, max=1)

        # Assert the channel dimension to be 1
        assert spatial_mask.shape[1] == 1

        return spatial_mask

    @staticmethod
    def _depth_mask(hr_depth: torch.Tensor, upsampled_lr_depth: torch.Tensor, threashold: float = 0.04):
        """Compute the depth mask.
        """
        depth_diff = torch.abs(hr_depth - upsampled_lr_depth)
        depth_diff = depth_diff - threashold

        depth_mask = torch.heaviside(depth_diff, values=torch.tensor(0.0))

        return depth_mask
    
    @staticmethod
    def _normal_mask(hr_normal: torch.Tensor, upsampled_lr_normal: torch.Tensor, threashold: float = 0.4):
        """Compute the normal mask.
        """
        # Dot product between the two normal maps
        normal_dotted = torch.sum(hr_normal * upsampled_lr_normal, dim=1, keepdim=True)

        # Subtraction
        normal_diff = threashold - normal_dotted
        normal_mask = torch.heaviside(normal_diff, values=torch.tensor(0.0))

        return normal_mask
    
    @staticmethod
    def _albedo_mask(hr_albedo: torch.Tensor, upsampled_lr_albedo: torch.Tensor, threashold: float = 0.1):
        """Compute the albedo mask.
        """

        albedo_diff = torch.abs(hr_albedo - upsampled_lr_albedo)
        albedo_diff = albedo_diff - threashold
        # Heaviside step function
        albedo_mask = torch.heaviside(albedo_diff, values=torch.tensor(0.0))
        # Convert the 3 channel mask to 1 channel
        albedo_mask = albedo_mask.sum(dim=1, keepdim=True)
        # Clamp the mask to 1
        albedo_mask = torch.clamp(albedo_mask, min=0, max=1)

        return albedo_mask
    
    @staticmethod
    def warp_frame(
        frame: torch.Tensor,
        motion_vector: torch.Tensor
    ) -> torch.Tensor:
        """Warp the frame using the motion vector.
        """
        # Scale motion vector, is ideally not needed but there is issue with our dataset
        # We had to increase the render resolution by 25% to acquire the correct resolution G-buffers
        # But this caused the motion vector to be 25% larger than the original frame
        motion_vector = motion_vector * 0.75

        n, c, h, w = frame.shape
        device = frame.device

        # Create normalized coordinate grid
        dx = torch.linspace(-1.0, 1.0, w, device=device)
        dy = torch.linspace(-1.0, 1.0, h, device=device)
        meshgrid: Tuple[torch.Tensor, torch.Tensor] = torch.meshgrid(dy, dx, indexing='ij')
        grid_y, grid_x = meshgrid

        # Add motion vectors to the grid
        grid_x = grid_x.unsqueeze(0).expand(n, -1, -1) - (2 * motion_vector[:, 0] / w)
        grid_y = grid_y.unsqueeze(0).expand(n, -1, -1) + (2 * motion_vector[:, 1] / h)
        warped_grid = torch.stack((grid_x, grid_y), dim=-1) # (N, H, W, 2)

        # Warp the image using grid_sample
        warped_frame = torch.nn.functional.grid_sample(frame, warped_grid, padding_mode='zeros', align_corners=True)
        return warped_frame
    
    @staticmethod
    def _temporal_mask(
        upsampled_base_color: torch.Tensor,
        warped_base_color: torch.Tensor,
        upsampled_depth: torch.Tensor,
        warped_depth: torch.Tensor,
        upsampled_normal: torch.Tensor,
        warped_normal: torch.Tensor
    ) -> torch.Tensor:
        """Compute the temporal masks.
        """

        depth_mask = torch.abs(upsampled_base_color - warped_base_color)
        albedo_mask = torch.abs(upsampled_depth - warped_depth)

        # Compute the cosine similarity between the normals
        cosine_similarity = torch.sum(upsampled_normal * warped_normal, dim=1, keepdim=True)
        cosine_similarity = torch.clamp(cosine_similarity, -1.0, 1.0)
        normal_mask = 1.0 - cosine_similarity # (B, 1, H, W)
        normal_mask = normal_mask / 2.0

        # Concatenate the masks
        temporal_mask = torch.cat([albedo_mask, depth_mask, normal_mask], dim=1)
        return temporal_mask

    @staticmethod
    def brdf_demodulate(
        frame: torch.Tensor,
        brdf_map: torch.Tensor
    ) -> torch.Tensor:
        """Demodulate the frame using the BRDF map.
        """

        pos = (brdf_map <= 1e-6)
        brdf_map[pos] = 1.0
        demodulated_frame = frame / brdf_map
        demodulated_frame[pos] = frame[pos]
        brdf_map[pos] = 0.0

        return demodulated_frame

    @staticmethod
    def brdf_remodulate(
        frame: torch.Tensor,
        brdf_map: torch.Tensor
    ) -> torch.Tensor:
        """Remodulate the frame using the BRDF map.
        """

        pos = (brdf_map <= 1e-6)
        brdf_map[pos] = 1.0
        remodulated_frame = frame * brdf_map
        brdf_map[pos] = 0.0

        return remodulated_frame
        