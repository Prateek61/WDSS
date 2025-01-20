# Losses and evaluation metrics for the network

import torch
import torch.nn as nn
# Torch ignite  
from utils.pytorch_ssim import ssim, SSIM

from lpips import LPIPS

from config import device

from typing import List, Dict, Tuple

class ImageEvaluator:
    lipps_model = LPIPS(net='vgg').to(device)

    @staticmethod
    def ssim(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Structural Similarity Index (SSIM) between two images.
        """
        return ssim(input, target)
    
    @staticmethod
    def psnr(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = torch.nn.functional.mse_loss(input, target)
        max_val: torch.Tensor = max(input.max(), target.max())
        return 20 * torch.log10(max_val / torch.sqrt(mse))
    
    @staticmethod
    def mse(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(input, target)

    @staticmethod
    def lpips(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Get the device of the input tensor
        input_device = input.device

        if input_device != device:
            input = input.to(device)
            target = target.to(device)

        # Compute the LPIPS value
        with torch.no_grad():
            lpips_val: torch.Tensor = ImageEvaluator.lipps_model(input, target)

        # Move the LPIPS value back to the input device
        if input_device != device:
            lpips_val = lpips_val.to(input_device)

        return lpips_val

class CriteronMSE(nn.Module):
    def __init__(self):
        super(CriteronMSE, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse(input, target)

class CriteronMSE_SSIM(nn.Module):
    def __init__(self, alpha: float = 0.8, channels: int = 3):
        super(CriteronMSE_SSIM, self).__init__()
        self.alpha = alpha
        self.channels = channels
        self.mse = nn.MSELoss().to(device)
        self.ssim = SSIM(channel=channels).to(device)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.mse(input, target) + (1 - self.alpha) * (1 - self.ssim(input, target))
    
class CriteronSSIM_LPIPS(nn.Module):
    def __init__(self, alpha: float = 0.6, beta: float = 0.4, channels: int = 3):
        super(CriteronSSIM_LPIPS, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.channels = channels
        self.ssim = SSIM(channel=channels).to(device)
        self.lpips = ImageEvaluator.lipps_model

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.alpha * (1 - self.ssim(input, target)) + self.beta * self.lpips(input, target)
    
class CriteronList(nn.Module):
    def __init__(self, criterions: List[nn.Module], weights: List[float]):
        super(CriteronList, self).__init__()
        self.criterions = criterions
        self.weights = weights

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = [weight * criterion(input, target) for criterion, weight in zip(self.criterions, self.weights)]
        return sum(loss)

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm, self).__init__()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(prediction - target)) / torch.numel(prediction)

class WaveletCriterion(nn.Module):
    def __init__(self):
        super(WaveletCriterion, self).__init__()
        self.lpips = LPIPS(net='vgg').to(device)
        self.l1 = L1Norm()
        self.ssim = SSIM().to(device)

    def forward(self, prediction_wavelet: torch.Tensor, 
                target_wavelet: torch.Tensor, 
                prediction_image: torch.Tensor, 
                target_image: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}

        # First compute all the losses for prediction and target image
        l1_loss = torch.mean(self.l1(prediction_image, target_image))
        ssim_loss = torch.mean(1 - self.ssim(prediction_image, target_image))
        lpips_loss = torch.mean(self.lpips(prediction_image, target_image))

        # Compute the L1 loss for the wavelet coefficients
        l1_loss_app = torch.mean(self.l1(prediction_wavelet[:, 0:3, :, :], target_wavelet[:, 0:3, :, :]))
        l1_loss_hor = torch.mean(self.l1(prediction_wavelet[:, 3:6, :, :], target_wavelet[:, 3:6, :, :]))
        l1_loss_ver = torch.mean(self.l1(prediction_wavelet[:, 6:9, :, :], target_wavelet[:, 6:9, :, :]))
        l1_loss_diag = torch.mean(self.l1(prediction_wavelet[:, 9:12, :, :], target_wavelet[:, 9:12, :, :]))

        # Compute the SSIM loss for the wavelet coefficients
        ssim_loss_app = torch.mean(1 - self.ssim(prediction_wavelet[:, 0:3, :, :], target_wavelet[:, 0:3, :, :]))
        ssim_loss_hor = torch.mean(1 - self.ssim(prediction_wavelet[:, 3:6, :, :], target_wavelet[:, 3:6, :, :]))
        ssim_loss_ver = torch.mean(1 - self.ssim(prediction_wavelet[:, 6:9, :, :], target_wavelet[:, 6:9, :, :]))
        ssim_loss_diag = torch.mean(1 - self.ssim(prediction_wavelet[:, 9:12, :, :], target_wavelet[:, 9:12, :, :]))

        # Compute the LPIPS loss for the wavelet coefficients
        lpips_loss_app = torch.mean(self.lpips(prediction_wavelet[:, 0:3, :, :], target_wavelet[:, 0:3, :, :]))

        # Compute the total loss
        total_loss = l1_loss + 0.2 * ssim_loss + 0.1 * lpips_loss
        total_loss += (l1_loss_app + l1_loss_hor + l1_loss_ver + l1_loss_diag) / 4
        total_loss += 0.2 * (ssim_loss_app + ssim_loss_hor + ssim_loss_ver + ssim_loss_diag) / 4
        total_loss += 0.05 * lpips_loss_app

        # Add all the losses to the dictionary
        losses['l1_loss'] = l1_loss
        losses['ssim_loss'] = ssim_loss
        losses['lpips_loss'] = lpips_loss
        losses['l1_loss_app'] = l1_loss_app
        losses['l1_loss_hor'] = l1_loss_hor
        losses['l1_loss_ver'] = l1_loss_ver
        losses['l1_loss_diag'] = l1_loss_diag
        losses['ssim_loss_app'] = ssim_loss_app
        losses['ssim_loss_hor'] = ssim_loss_hor
        losses['ssim_loss_ver'] = ssim_loss_ver
        losses['ssim_loss_diag'] = ssim_loss_diag
        losses['lpips_loss_app'] = lpips_loss_app
        losses['total_loss'] = total_loss

        return total_loss, losses

        