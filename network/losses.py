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

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(prediction - target)) / torch.numel(prediction)

class CriterionSimple(CriterionBase):
    """
    Simple L1 loss function.
    """
    def __init__(self, wavelet_weigt: float, image_weights: float) -> None:
        super(CriterionSimple, self).__init__("Simple")
        self.l1 = L1Norm()
        self.wavelet_weight = wavelet_weigt
        self.image_weight = image_weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_wavelets: torch.Tensor,
        target_wavelets: torch.Tensor,
        inps: Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        loss_image = self.l1(pred, target)
        loss_wavelet = self.l1(pred_wavelets, target_wavelets)
        loss = self.image_weight * loss_image + self.wavelet_weight * loss_wavelet
        return loss, {
            "image_l1": loss_image,
            "wavelet_l1": loss_wavelet
        }
    

class CriterionSimpleNorm(CriterionBase):
    """
    Simple L1 loss function with normalization.
    """
    def __init__(self, wavelet_weight: float, image_weights: float) -> None:
        super(CriterionSimpleNorm, self).__init__("SimpleNorm")
        self.l1 = L1Norm()
        self.wavelet_weight = wavelet_weight
        self.image_weight = image_weights

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_wavelets: torch.Tensor,
        target_wavelets: torch.Tensor,
        inps: Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_diff = torch.abs(pred - target)
        wavelet_diff = torch.abs(pred_wavelets - target_wavelets)

        # Normalization
        image_diff = reinhard_norm(image_diff)
        wavelet_diff = reinhard_norm(wavelet_diff)

        loss_image = torch.sum(image_diff) / torch.numel(image_diff)
        loss_wavelet = torch.sum(wavelet_diff) / torch.numel(wavelet_diff)

        loss = self.image_weight * loss_image + self.wavelet_weight * loss_wavelet

        return loss, {
            "image_l1": loss_image,
            "wavelet_l1": loss_wavelet,
            "image_l1_no_norm": self.l1(pred, target),
            "wavelet_l1_no_norm": self.l1(pred_wavelets, target_wavelets)
        }
