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


# class Criterion_Combined(CriterionBase):
#     def __init__(self, weights: Dict[str, float] = {
#         "l1": 0.25,
#         "ssim": 0.2,
#         "l1_wave": 0.25,
#         "ssim_reconstructed": 0.2,
#         "l1_reconstructed": 0.6,
#         "lpips_reconstructed": 0.15
#     }):
#         super(Criterion_Combined, self).__init__()
#         self.lpips = LPIPS(net='alex')
#         self.l1 = L1Norm()
#         self.ssim = SSIM()

#         self.weights = weights

#     def forward(self, prediction_wavelet: torch.Tensor, 
#                 target_wavelet: torch.Tensor, 
#                 pred: torch.Tensor, 
#                 target: torch.Tensor,
#                 extra : Dict[str, torch.Tensor] = {}
#                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         losses: Dict[str, torch.Tensor] = {}

#         # First compute all the losses for prediction and target image
#         l1_loss = self.l1.forward(pred, target)
            
#         # Compute the L1 loss for the wavelet coefficients
#         l1_wave = self.l1.forward(prediction_wavelet, target_wavelet)

#         wave_min = torch.min(target_wavelet.min(), prediction_wavelet.min())
#         target_wavelet = target_wavelet - wave_min
#         prediction_wavelet = prediction_wavelet - wave_min

#         # prediction_image = reinhard_norm(prediction_image)
#         # target_image = reinhard_norm(target_image)

#         ssim_loss = 1 - self.ssim.forward(pred, target)

#         ssim_reconstructed = 1 - self.ssim.forward(extra['img_processed'], extra['hr_processed'])
#         l1_reconstructed = self.l1.forward(extra['img_processed'], extra['hr_processed'])
#         lpips_reconstructed = self.lpips(extra['img_processed']*2 -1, extra['hr_processed']*2 -1).mean()

#         # Compute the total loss
#         total_loss = self.weights['l1'] * l1_loss + self.weights['ssim'] * ssim_loss + self.weights['l1_wave'] * l1_wave + self.weights['ssim_reconstructed'] * ssim_reconstructed + self.weights['l1_reconstructed'] * l1_reconstructed + self.weights['lpips_reconstructed'] * lpips_reconstructed

#         # Add all the losses to the dictionary
#         losses['l1_image'] = l1_loss
#         losses['ssim_image'] = ssim_loss
#         losses['lpips_reconstructed'] = lpips_reconstructed
#         losses['l1_reconstructed'] = l1_reconstructed
#         losses['ssim_reconstructed'] = ssim_reconstructed
#         losses['l1_wavelets'] = l1_wave

#         if total_loss == float('nan'):
#             # Print all the losses
#             print(f"L1 Loss Image: {l1_loss}")
#             print(f"SSIM Loss Image: {ssim_loss}")
#             print(f"L1 Loss Wavelet: {l1_wave}")
#             print(f"L1 Loss Reconstructed: {l1_reconstructed}")
#             print(f"SSIM Loss Reconstructed: {ssim_reconstructed}")
#             print(f"LPIPS Loss Reconstructed: {lpips_reconstructed}")
#             print(f"Total Loss: {total_loss}")
#             total_loss = torch.tensor(0.0)

#         return total_loss , losses

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
        self.lpips_model.eval()
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
