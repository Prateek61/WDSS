# Losses and evaluation metrics for the network

import torch
import torch.nn as nn
# Torch ignite  
from utils.pytorch_ssim import ssim, SSIM

from lpips import LPIPS
from torchvision.models import vgg16, VGG16_Weights

from config import device

from typing import List

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