from abc import ABC, abstractmethod
import torch
from torch import nn
from utils.pytorch_ssim import ssim, SSIM
from config import device
from .image_evaluator import exp_norm, reinhard_norm
from lpips import LPIPS
from utils.preprocessor import Preprocessor

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

class CriterionOld(CriterionBase):
    def __init__(self,
        preprocessor: Preprocessor,
        l1_irridiance: float = 0.25,
        l1_wavelet: float = 0.25,
        l1_reconstructed: float = 0.6,
        ssim_reconstructed: float = 0.2,
        lpips_reconstructed: float = 0.15
    ):
        super(CriterionOld, self).__init__("Combined")
        self.lpips_model = LPIPS(net='alex')
        # self.lpips_model.eval()
        self.l1_loss = L1Norm()
        self.ssim_loss = SSIM()
        self.preprocessor = preprocessor

        self.l1_irridiance_weight = l1_irridiance
        self.l1_wavelet_weight = l1_wavelet
        self.l1_reconstructed_weight = l1_reconstructed
        self.ssim_reconstructed_weight = ssim_reconstructed
        self.lpips_reconstructed_weight = lpips_reconstructed

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_wavelets: torch.Tensor,
        target_wavelets: torch.Tensor,
        inps: Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        metrics: Dict[str, torch.Tensor] = {}

        # Compute the L1 loss for the wavelet coefficients
        l1_wavelet = self.l1_loss(pred_wavelets, target_wavelets)
        metrics["l1_wavelet"] = l1_wavelet

        # Compute the L1 loss for the irradiance image
        l1_irridiance = self.l1_loss(pred, target)
        metrics["l1_irridiance"] = l1_irridiance

        # Post-process the frames
        pred_processed = self.preprocessor.postprocess(pred, inps["EXTRA"])
        target_processed = self.preprocessor.postprocess(target, inps["EXTRA"])

        # Compute the L1 loss for the reconstructed image
        l1_reconstructed = self.l1_loss(pred_processed, target_processed)
        metrics["l1_reconstructed"] = l1_reconstructed

        # Compute the SSIM loss for the reconstructed image
        ssim_reconstructed = 1 - self.ssim_loss(pred_processed, target_processed)
        metrics["ssim_reconstructed"] = ssim_reconstructed

        # Compute the LPIPS loss for the reconstructed image
        pred_processed = reinhard_norm(pred_processed)
        target_processed = reinhard_norm(target_processed)
        lpips_reconstructed = self.lpips_model(pred_processed * 2 - 1, target_processed * 2 - 1).mean()
        metrics["lpips_reconstructed"] = lpips_reconstructed

        # Compute the total loss
        total_loss = (
            self.l1_irridiance_weight * l1_irridiance +
            self.l1_wavelet_weight * l1_wavelet +
            self.l1_reconstructed_weight * l1_reconstructed +
            self.ssim_reconstructed_weight * ssim_reconstructed +
            self.lpips_reconstructed_weight * lpips_reconstructed
        )

        if total_loss != total_loss:  # Check for NaN
            total_loss = torch.tensor(0.0, device=pred.device)
            print("Total loss is NaN. Setting to 0.0")

        return total_loss, metrics

class CriterionDFASR(CriterionBase):
    def __init__(
        self,
        preprocessor: Preprocessor,
        l1_wavelet: float = 0.1,
        l1_reconstructed: float = 0.1,
        ssim_reconstructed: float = 0.4,
        perceptual_reconstructed: float = 0.2,
        mask: float = 0.1,
        temporal: float = 0.3,
        lpips_net: str = 'alex'
    ):
        super(CriterionDFASR, self).__init__("DFASR")
        self.lpips_model = LPIPS(net=lpips_net)
        self.l1_loss = L1Norm()
        self.ssim_loss = SSIM()
        self.preprocessor = preprocessor

        self.l1_wavelet_weight = l1_wavelet
        self.l1_reconstructed_weight = l1_reconstructed
        self.ssim_reconstructed_weight = ssim_reconstructed
        self.perceptual_reconstructed_weight = perceptual_reconstructed
        self.mask_weight = mask
        self.temporal_weight = temporal
        self.lpips_net = lpips_net

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        pred_wavelets: torch.Tensor,
        target_wavelets: torch.Tensor,
        inps: Dict[str, torch.Tensor | Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        extra = inps["EXTRA"]
        metrics: Dict[str, torch.Tensor] = {}

        # Compute the L1 loss for the wavelet coefficients
        l1_wavelet = self.l1_loss(pred_wavelets, target_wavelets)
        metrics["l1_wavelet"] = l1_wavelet

        # Post-process the frames
        pred_processed = self.preprocessor.postprocess(pred, extra)
        target_processed = self.preprocessor.postprocess(target, extra)

        # Compute the L1 loss for the reconstructed image
        l1_reconstructed = self.l1_loss(pred_processed, target_processed)
        metrics["l1_reconstructed"] = l1_reconstructed
        # Compute the SSIM loss for the reconstructed image
        ssim_reconstructed = 1 - self.ssim_loss(pred_processed, target_processed)
        metrics["ssim_reconstructed"] = ssim_reconstructed
        # Compute the LPIPS loss for the reconstructed image
        pred_processed_norm = reinhard_norm(pred_processed)
        target_processed_norm = reinhard_norm(target_processed)
        lpips_reconstructed = self.lpips_model(pred_processed_norm * 2 - 1, target_processed_norm * 2 - 1).mean()
        metrics["lpips_reconstructed"] = lpips_reconstructed

        # Compute the mask loss
        spatial_mask = extra["SPATIAL_MASK"]
        mask_loss = (spatial_mask * torch.abs(pred_processed - target_processed)).sum() / (spatial_mask.sum() + 1)
        metrics["mask_loss"] = mask_loss

        # Temporal loss
        temporal_pretonemap_warped = extra["TEMPORAL_PRETONEMAP"]
        temporal_output = pred_processed - temporal_pretonemap_warped
        temporal_target = target_processed - temporal_pretonemap_warped
        temporal_loss = self.l1_loss(temporal_output, temporal_target)
        metrics["temporal_loss"] = temporal_loss

        # Compute the total loss
        total_loss = (
            self.l1_wavelet_weight * l1_wavelet +
            self.l1_reconstructed_weight * l1_reconstructed +
            self.ssim_reconstructed_weight * ssim_reconstructed +
            self.perceptual_reconstructed_weight * lpips_reconstructed +
            self.mask_weight * mask_loss +
            self.temporal_weight * temporal_loss
        )
        if total_loss != total_loss:
            total_loss = torch.tensor(0.0, device=pred.device)
            print("Total loss is NaN. Setting to 0.0")
            print(metrics)
        return total_loss, metrics
    