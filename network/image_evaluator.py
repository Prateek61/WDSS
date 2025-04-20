import torch
import torch.nn as nn
from utils.pytorch_ssim import ssim, SSIM

from lpips import LPIPS
from config import device
from typing import List, Dict, Tuple, Optional

def exp_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Exponential normalization of the input tensor.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Normalized tensor.
    """
    return 1.0 - torch.exp(-x)  # Normalize to [0, 1]

def reinhard_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Reinhard normalization of the input tensor.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Normalized tensor.
    """
    return x / (x + 1.0)  # Normalize to [0, 1]


class ImageEvaluator(nn.Module):
    """
    Class to evaluate image quality using various metrics.
    """
    lpips_model: Optional[LPIPS] = None
    _lpips_device: Optional[torch.device] = None

    @staticmethod
    def initialize_lpips(lpips_device: torch.device = device) -> None:
        """
        Initialize the LPIPS model.
        Args:
            lpips_device (torch.device): Device to run the LPIPS model on.
        """
        if ImageEvaluator.lpips_model is None or ImageEvaluator._lpips_device != lpips_device:
            ImageEvaluator.lpips_model = LPIPS(net='vgg').to(lpips_device)
            ImageEvaluator._lpips_device = lpips_device

    @staticmethod
    def ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the SSIM between two images.
        Args:
            img1 (torch.Tensor): First image tensor.
            img2 (torch.Tensor): Second image tensor.
        Returns:
            torch.Tensor: SSIM value.
        """
        return ssim(img1, img2)
    
    @staticmethod
    def psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float) -> torch.Tensor:
        """
        Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
        Args:
            img1 (torch.Tensor): First image tensor.
            img2 (torch.Tensor): Second image tensor.
        Returns:
            torch.Tensor: PSNR value.
        """
        mse = torch.nn.functional.mse_loss(img1, img2)
        return 20 * torch.log10(max_val / torch.sqrt(mse))
    
    @staticmethod
    def mse(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Mean Squared Error (MSE) between two images.
        Args:
            img1 (torch.Tensor): First image tensor.
            img2 (torch.Tensor): Second image tensor.
        Returns:
            torch.Tensor: MSE value.
        """
        return torch.nn.functional.mse_loss(img1, img2)
    
    @staticmethod
    def lpips(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the Learned Perceptual Image Patch Similarity (LPIPS) between two images.
        Args:
            img1 (torch.Tensor): First image tensor.
            img2 (torch.Tensor): Second image tensor.
        Returns:
            torch.Tensor: LPIPS value.
        """
        if ImageEvaluator.lpips_model is None:
            ImageEvaluator.initialize_lpips()
        
        input_device = input.device

        if input_device != ImageEvaluator._lpips_device:
            img1 = img1.to(ImageEvaluator._lpips_device)
            img2 = img2.to(ImageEvaluator._lpips_device)

        with torch.no_grad():
            lpips_value = ImageEvaluator.lpips_model((img1 * 2) - 1, (img2 * 2) - 1)

        if input_device != ImageEvaluator._lpips_device:
            lpips_value = lpips_value.to(input_device)

        return lpips_value

    @staticmethod
    def l1(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Calculate the L1 loss between two images.
        Args:
            img1 (torch.Tensor): First image tensor.
            img2 (torch.Tensor): Second image tensor.
        Returns:
            torch.Tensor: L1 loss value.
        """
        return torch.nn.functional.l1_loss(img1, img2)