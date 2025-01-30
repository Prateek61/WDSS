# Abstract base class for a model
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

from typing import Tuple, List

class ModelBase(nn.Module, ABC):
    def __init__(self):
        super(ModelBase, self).__init__()

    @abstractmethod
    def forward(self, lr_frame: torch.Tensor, hr_gbuffer: torch.Tensor, temporal: torch.Tensor, upscale_factor: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model

        Args:
            lr_frame (torch.Tensor): Low resolution frame
            hr_frame (torch.Tensor): High resolution g-buffers and spatial masks
            temporal (torch.Tensor): Temporal frame and temporal masks

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output wavelet coefficients and final image
        """
        pass
