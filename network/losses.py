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

        # Weights for different losses
        weights = {
            'l1': 1.0,       # L1 loss has the highest weight
            'ssim': 0.2,     # SSIM loss contributes moderately
            'lpips': 0.1,    # LPIPS loss has the smallest weight
        }

        # L1 loss for the wavelet coefficients

        l1_loss = torch.mean(torch.abs(prediction_wavelet - target_wavelet))
        ssim_loss = torch.mean(1 - self.ssim(prediction_image, target_image))
        lpips_loss = torch.mean(self.lpips(prediction_image, target_image))
        
        total_loss = weights['l1'] * l1_loss + weights['ssim'] * ssim_loss + weights['lpips'] * lpips_loss
        
        losses['l1'] = l1_loss
        losses['ssim'] = ssim_loss
        losses['lpips'] = lpips_loss
        
        

        # Return total loss and individual losses
        return total_loss, losses

