from abc import ABC, abstractmethod
import torch
from torch import nn
from utils.pytorch_ssim import ssim, SSIM
from config import device
from .image_evaluator import exp_norm, reinhard_norm

from typing import List, Dict, Tuple, Optional

class CriterionBase(nn.Module, ABC):
    """
    Base class for loss functions.
    """
    def __init__(self, name: str) -> None:
        super(CriterionBase, self).__init__()
        self.name = name

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_wavelets: torch.Tensor,
        target_wavelets: torch.Tensor,
        inps: Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Abstract method to compute the loss.
        """
        pass

