# Losses and evaluation metrics for the network
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
# Torch ignite  
from utils.pytorch_ssim import ssim, SSIM

from lpips import LPIPS

from config import device

from typing import List, Dict, Tuple

def exp_norm(x: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.exp(-x)

def reinhard_norm(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + x)

class ImageEvaluator:
    lipps_model = None

    @staticmethod
    def _setup_lpips():
        if ImageEvaluator.lipps_model is None:
            ImageEvaluator.lipps_model = LPIPS(net='vgg').to(device)

    @staticmethod
    def ssim(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Structural Similarity Index (SSIM) between two images.
        """
        return ssim(input, target)
    
    @staticmethod
    def psnr(input: torch.Tensor, target: torch.Tensor, max_val: float) -> torch.Tensor:
        mse = torch.nn.functional.mse_loss(input, target)
        return 20 * torch.log10(max_val / torch.sqrt(mse))
    
    @staticmethod
    def mse(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(input, target)

    @staticmethod
    def lpips(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ImageEvaluator._setup_lpips()

        # Get the device of the input tensor
        input_device = input.device

        if input_device != device:
            input = input.to(device)
            target = target.to(device)

        # Compute the LPIPS value
        with torch.no_grad():
            lpips_val: torch.Tensor = ImageEvaluator.lipps_model((input * 2) - 1, (target * 2) - 1)

        # Move the LPIPS value back to the input device
        if input_device != device:
            lpips_val = lpips_val.to(input_device)

        return lpips_val
    
    @staticmethod
    def l1(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.abs(input - target)) / torch.numel(input)


# Abstract base class for criterion
class CriterionBase(nn.Module, ABC):
    def __init__(self):
        super(CriterionBase, self).__init__()

    @abstractmethod
    def forward(self, predicted_wavelets: torch.Tensor, target_wavelets: torch.Tensor, predicted_image: torch.Tensor, target_image: torch.Tensor, extra : Dict[str, torch.Tensor] = {}) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of the criterion

        Args:
            predicted_wavelets (torch.Tensor): Predicted wavelet coefficients
            target_wavelets (torch.Tensor): Target wavelet coefficients
            predicted_image (torch.Tensor): Predicted image
            target_image (torch.Tensor): Target image

        Returns:
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]: Total loss and dictionary of individual losses
        """
        pass

class CriterionMSE(CriterionBase):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5):
        """ Mean Squared Error (MSE) criterion

        Args:
            alpha (float, optional): Weight for the predicted image loss. Defaults to 0.5.
            beta (float, optional): Weight for the wavelet coefficients loss. Defaults to 0.5.
        """
        super(CriterionMSE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, predicted_wavelets: torch.Tensor, target_wavelets: torch.Tensor, predicted_image: torch.Tensor, target_image: torch.Tensor, extra : Dict[str, torch.Tensor] = {}) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Compute the MSE loss for the predicted image
        mse_image = self.mse.forward(predicted_image, target_image)
        mse_wavelets = self.mse.forward(predicted_wavelets, target_wavelets)

        # Compute the total loss
        total_loss = self.alpha * mse_image + self.beta * mse_wavelets

        # Return the total loss and the individual losses
        return total_loss, {'mse_image': mse_image, 'mse_wavelets': mse_wavelets, 'total_loss': total_loss}
    
class CriterionSSIM_MSE(CriterionBase):
    def __init__(self, weights: Dict[str, float] = {}):
        super(CriterionSSIM_MSE, self).__init__()
        if not weights:
            self.weights = {
                'ssim_image': 0.1,
                'ssim_wavelets': 0.1,
                'mse_image': 0.5,
                'mse_wavelets': 0.5
            }
        else:  
            self.weights = weights

        self.ssim = SSIM().to(device)
        self.mse = nn.MSELoss().to(device)

    def forward(self, predicted_wavelets: torch.Tensor, target_wavelets: torch.Tensor, predicted_image: torch.Tensor, target_image: torch.Tensor, extra : Dict[str, torch.Tensor] = {}) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Compute the SSIM loss for the predicted image
        ssim_image = 1 - self.ssim.forward(predicted_image, target_image)
        ssim_approx = 1 - self.ssim.forward(predicted_wavelets[:, 0:3, :, :], target_wavelets[:, 0:3, :, :])
        ssim_horizontal = 1 - self.ssim.forward(predicted_wavelets[:, 3:6, :, :], target_wavelets[:, 3:6, :, :])
        ssim_vertical = 1 - self.ssim.forward(predicted_wavelets[:, 6:9, :, :], target_wavelets[:, 6:9, :, :])
        ssim_diagonal = 1 - self.ssim.forward(predicted_wavelets[:, 9:12, :, :], target_wavelets[:, 9:12, :, :])
        ssim_wavelets = (ssim_approx + ssim_horizontal + ssim_vertical + ssim_diagonal) / 4

        # Compute the MSE loss for the predicted image
        mse_image = self.mse.forward(predicted_image, target_image)
        mse_wavelets = self.mse.forward(predicted_wavelets, target_wavelets)

        # Compute the total loss
        total_loss = self.weights['ssim_image'] * ssim_image + self.weights['ssim_wavelets'] * ssim_wavelets + self.weights['mse_image'] * mse_image + self.weights['mse_wavelets'] * mse_wavelets

        # Return the total loss and the individual losses
        return total_loss, {'ssim_image': ssim_image, 'ssim_wavelets': ssim_wavelets, 'mse_image': mse_image, 'mse_wavelets': mse_wavelets, 'total_loss': total_loss}

      
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
    
class CriterionSSIM_L1(CriterionBase):
    def __init__(self, weights: Dict[str, float] = {
        'ssim_image': 0.1,
        'ssim_wavelets': 0.1,
        'l1_image': 0.5,
        'l1_wavelets': 0.5
    }):
        super(CriterionSSIM_L1, self).__init__()
        self.weights = weights

        self.ssim = SSIM().to(device)
        self.l1 = L1Norm().to(device)

    def forward(self, predicted_wavelets: torch.Tensor, target_wavelets: torch.Tensor, predicted_image: torch.Tensor, target_image: torch.Tensor, extra : Dict[str, torch.Tensor] = {}) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Compute the SSIM loss for the predicted image
        ssim_image = 1 - self.ssim.forward(predicted_image, target_image)
        ssim_approx = 1 - self.ssim.forward(predicted_wavelets[:, 0:3, :, :], target_wavelets[:, 0:3, :, :])
        ssim_horizontal = 1 - self.ssim.forward(predicted_wavelets[:, 3:6, :, :], target_wavelets[:, 3:6, :, :])
        ssim_vertical = 1 - self.ssim.forward(predicted_wavelets[:, 6:9, :, :], target_wavelets[:, 6:9, :, :])
        ssim_diagonal = 1 - self.ssim.forward(predicted_wavelets[:, 9:12, :, :], target_wavelets[:, 9:12, :, :])
        ssim_wavelets = (ssim_approx + ssim_horizontal + ssim_vertical + ssim_diagonal) / 4

        # Compute the L1 loss for the predicted image
        l1_image = torch.mean(self.l1(predicted_image, target_image))
        l1_wavelets = torch.mean(self.l1(predicted_wavelets, target_wavelets))

         # Compute the total loss
        total_loss = self.weights['ssim_image'] * ssim_image + self.weights['ssim_wavelets'] * ssim_wavelets + self.weights['l1_image'] * l1_image + self.weights['l1_wavelets'] * l1_wavelets

        # Return the total loss and the individual losses
        return total_loss, {'ssim_image': ssim_image, 'ssim_wavelets': ssim_wavelets, 'l1_image': l1_image, 'l1_wavelets': l1_wavelets}


class CriterionSSIM_SmooothL1(CriterionBase):
    def __init__ (self, weights: Dict[str, float] = {
        'ssim_image': 0.1,
        'ssim_wavelets': 0.1,
        'smooth_l1_image': 0.5,
        'smooth_l1_wavelets': 0.5
    }):
        super(CriterionSSIM_SmooothL1, self).__init__()
        self.weights = weights

        self.ssim = SSIM().to(device)
        self.smooth_l1 = nn.SmoothL1Loss().to(device)

    def forward(self, predicted_wavelets: torch.Tensor, target_wavelets: torch.Tensor, predicted_image: torch.Tensor, target_image: torch.Tensor, extra : Dict[str, torch.Tensor] = {}) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        #compute the SSIM loss for the predicted image
        ssim_image = 1 - self.ssim.forward(predicted_image, target_image)
        ssim_approx = 1 - self.ssim.forward(predicted_wavelets[:, 0:3, :, :], target_wavelets[:, 0:3, :, :])
        ssim_horizontal = 1 - self.ssim.forward(predicted_wavelets[:, 3:6, :, :], target_wavelets[:, 3:6, :, :])
        ssim_vertical = 1 - self.ssim.forward(predicted_wavelets[:, 6:9, :, :], target_wavelets[:, 6:9, :, :])
        ssim_diagonal = 1 - self.ssim.forward(predicted_wavelets[:, 9:12, :, :], target_wavelets[:, 9:12, :, :])
        ssim_wavelets = (ssim_approx + ssim_horizontal + ssim_vertical + ssim_diagonal) / 4

        #compute the smooth L1 loss for the predicted image
        smooth_l1_image = self.smooth_l1.forward(predicted_image, target_image)
        smooth_l1_wavelets = self.smooth_l1.forward(predicted_wavelets, target_wavelets)

        #compute the total loss
        total_loss = self.weights['ssim_image'] * ssim_image + self.weights['ssim_wavelets'] * ssim_wavelets + self.weights['smooth_l1_image'] * smooth_l1_image + self.weights['smooth_l1_wavelets'] * smooth_l1_wavelets

        #Return the total loss and the individual losses
        return total_loss, {'ssim_image': ssim_image, 'ssim_wavelets': ssim_wavelets, 'smooth_l1_image': smooth_l1_image, 'smooth_l1_wavelets': smooth_l1_wavelets}


class Criterion_Combined(nn.Module):
    def __init__(self, weights: Dict[str, float] = {
        "l1": 0.25,
        "ssim": 0.2,
        "l1_wave": 0.25,
        "ssim_reconstructed": 0.2,
        "l1_reconstructed": 0.6,
        "lpips_reconstructed": 0.15
    }):
        super(Criterion_Combined, self).__init__()
        self.lpips = LPIPS(net='alex')
        self.l1 = L1Norm()
        self.ssim = SSIM()

        self.weights = weights

    def forward(self, prediction_wavelet: torch.Tensor, 
                target_wavelet: torch.Tensor, 
                prediction_image: torch.Tensor, 
                target_image: torch.Tensor,
                extra : Dict[str, torch.Tensor] = {}
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses: Dict[str, torch.Tensor] = {}

        # First compute all the losses for prediction and target image
        l1_loss = self.l1.forward(prediction_image, target_image)
            
        # Compute the L1 loss for the wavelet coefficients
        l1_wave = self.l1.forward(prediction_wavelet, target_wavelet)

        wave_min = torch.min(target_wavelet.min(), prediction_wavelet.min())
        target_wavelet = target_wavelet - wave_min
        prediction_wavelet = prediction_wavelet - wave_min

        # prediction_image = reinhard_norm(prediction_image)
        # target_image = reinhard_norm(target_image)

        ssim_loss = 1 - self.ssim.forward(prediction_image, target_image)

        ssim_reconstructed = 1 - self.ssim.forward(extra['img_processed'], extra['hr_processed'])
        l1_reconstructed = self.l1.forward(extra['img_processed'], extra['hr_processed'])
        lpips_reconstructed = self.lpips(extra['img_processed']*2 -1, extra['hr_processed']*2 -1).mean()

        # Compute the total loss
        total_loss = self.weights['l1'] * l1_loss + self.weights['ssim'] * ssim_loss + self.weights['l1_wave'] * l1_wave + self.weights['ssim_reconstructed'] * ssim_reconstructed + self.weights['l1_reconstructed'] * l1_reconstructed + self.weights['lpips_reconstructed'] * lpips_reconstructed

        # Add all the losses to the dictionary
        losses['l1_image'] = l1_loss
        losses['ssim_image'] = ssim_loss
        losses['lpips_reconstructed'] = lpips_reconstructed
        losses['l1_reconstructed'] = l1_reconstructed
        losses['ssim_reconstructed'] = ssim_reconstructed
        losses['l1_wavelets'] = l1_wave

        if total_loss == float('nan'):
            # Print all the losses
            print(f"L1 Loss Image: {l1_loss}")
            print(f"SSIM Loss Image: {ssim_loss}")
            print(f"L1 Loss Wavelet: {l1_wave}")
            print(f"L1 Loss Reconstructed: {l1_reconstructed}")
            print(f"SSIM Loss Reconstructed: {ssim_reconstructed}")
            print(f"LPIPS Loss Reconstructed: {lpips_reconstructed}")
            print(f"Total Loss: {total_loss}")
            total_loss = torch.tensor(0.0)

        return total_loss , losses

