import torch
import torch.nn as nn

from typing import List, Dict, Any

class BRDFProcessor:
    @staticmethod
    def compute_brdf(
        diffuse: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        specular: torch.Tensor,
        NoV: torch.Tensor,
        precomp: torch.Tensor,
        max_idx: int = 255
    ) -> torch.Tensor:
        nov_idx = (NoV * max_idx).long().clamp(0, max_idx)
        roughness_idx = (roughness * max_idx).long().clamp(0, max_idx)

        # Sample the pre-computed BRDF lookup table
        pre_integration = precomp[0][nov_idx, roughness_idx]
        pre_integration_b = precomp[1][nov_idx, roughness_idx]

        # Compute specular reflactance
        specular = specular.expand(3, -1, -1)
        metallic = metallic.expand(3, -1, -1)

        # Compute the specular reflactance
        specular_reflectance = torch.lerp(0.08 * specular, diffuse, metallic)

        # Calculate the BRDF
        brdf = diffuse * (1 - metallic) + specular_reflectance * pre_integration + pre_integration_b.expand(3, -1, -1)

        return brdf
    
    @staticmethod
    def exponential_normalize(frame: torch.Tensor, exposure: float = 1.0) -> torch.Tensor:
        return 1.0 - torch.exp(-frame * exposure)
    
    @staticmethod
    def exponential_denormalize(frame: torch.Tensor, exposure: float = 1.0) -> torch.Tensor:
        return -torch.log(1.0 - frame) / exposure
    
    @staticmethod
    def brdf_demodulate(
        frame: torch.Tensor,
        brdf_map: torch.Tensor
    ) -> torch.Tensor:
        demodulated_frame = torch.where(brdf_map < 0.004, 0, frame / brdf_map)
        return demodulated_frame
    
    @staticmethod
    def brdf_remodulate(
        frame: torch.Tensor,
        brdf_map: torch.Tensor
    ) -> torch.Tensor:
        remodulated_frame = torch.where(brdf_map < 0.004, 0, frame * brdf_map)
        return remodulated_frame
