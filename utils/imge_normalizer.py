import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from typing import List, Dict, Any

class BaseImageNormalizer(ABC):
    def __init__(self):
        ...

    @abstractmethod
    def normalize(self, x: torch.Tensor):
        raise NotImplementedError
    
    @abstractmethod
    def denormalize(self, x: torch.Tensor):
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
    