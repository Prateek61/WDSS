import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from typing import List, Dict, Any

class BaseImageNormalizer(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'BaseImageNormalizer':
        if config['type'] == 'log':
            return LogScaleNormalizer()
        elif config['type'] == 'exposure':
            return ExposureNormalizer(config['value'])
        elif config['type'] == 'exponential':
            return ExponentialNormalizer()
        elif config['type'] == 'srgb':
            return LinearToSRGB()
        elif config['type'] == 'auto_exposure':
            return AutoExposureNormalizer(config['percentile'])
        else:
            raise ValueError(f"Unknown normalizer type: {config['type']}")
    
class LogScaleNormalizer(BaseImageNormalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, x: torch.Tensor):
        return torch.log(x + 1.0)
    
    def denormalize(self, x: torch.Tensor):
        return torch.exp(x) - 1.0
    
    @staticmethod
    def from_config(config: Dict[str, Any]):
        return LogScaleNormalizer()
    
class ExposureNormalizer(BaseImageNormalizer):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def normalize(self, x: torch.Tensor):
        return x / self.value
    
    def denormalize(self, x: torch.Tensor):
        return x * self.value
    
    @staticmethod
    def from_config(config: Dict[str, Any]):
        return ExposureNormalizer(config['value'])
    
class ExponentialNormalizer(BaseImageNormalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, x: torch.Tensor):
        return 1.0 - torch.exp(-x)
    
    def denormalize(self, x: torch.Tensor):
        return -torch.log(1.0 - x)
    
    @staticmethod
    def from_config(config: Dict[str, Any]):
        return ExponentialNormalizer()
    
class LinearToSRGB(BaseImageNormalizer):
    def __init__(self):
        super().__init__()

    def normalize(self, x: torch.Tensor):
        return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1/2.4)) - 0.055)
    
        
    def denormalize(self, x: torch.Tensor):
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    
    @staticmethod
    def from_config(config: Dict[str, Any]):
        return LinearToSRGB()
    
class AutoExposureNormalizer(BaseImageNormalizer):
    def __init__(self, percentile: float):
        super().__init__()
        self.percentile = percentile

    def _luminance(self, x: torch.Tensor):
        return 0.2126 * x[0, :, :] + 0.7152 * x[1, :, :] + 0.0722 * x[2, :, :]

    def _exposure(self, x: torch.Tensor) -> torch.Tensor:
        luminance = self._luminance(x)
        with torch.no_grad():
            histogram, bin_edges = torch.histogram(luminance, bins=256, range=(0.0, 1.0))
            cumulative_histogram = torch.cumsum(histogram, dim=0)
            total_pixels = cumulative_histogram[-1]
            threashold = total_pixels * self.percentile
            bin_index = torch.searchsorted(cumulative_histogram, threashold)
            mean_luminance = bin_edges[bin_index]
            exposure = 0.18 / (mean_luminance + (1.0 - 0.18))
            return exposure
        
    def normalize(self, x: torch.Tensor):
        exposure = self._exposure(x).to(x.device)
        return x / exposure

    def denormalize(self, x: torch.Tensor):
        return x     
    