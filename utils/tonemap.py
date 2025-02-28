import torch
import torch.nn as nn

from typing import List, Dict, Any

class BaseTonemapper(nn.Module):
    def __init__(self):
        super(BaseTonemapper, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
    
    @staticmethod
    def from_name(name: str = 'Reinhard') -> 'BaseTonemapper':
        try:
            name, gain = name.split('-')
            gain = float(gain)
        except:
            gain = None

        tonemapper_class: type = BaseTonemapper

        if name == 'Reinhard':
            tonemapper_class = ReinhardTonemapper
        elif name == 'Hable':
            tonemapper_class = HableTonemapper
        elif name == 'SRGBHable':
            tonemapper_class = SRGBHable
        elif name == 'SRGB':
            tonemapper_class = SRGBTonemapper
        elif name == 'Exp':
            tonemapper_class = Exp
        elif name == 'Log':
            tonemapper_class = Log
        elif name == 'Default':
            tonemapper_class = DefaultTonemapper
        
        else:
            raise ValueError(f"Tonemapper {name} not supported.")
        
        if gain is not None:
            return tonemapper_class(gain)
        else:
            return tonemapper_class()
    
class ReinhardTonemapper(BaseTonemapper):
    def __init__(self, gain: float = 1.0):
        super(ReinhardTonemapper, self).__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gain
        res =  x / (1.0 + x)
        return res.clamp(0.0, 1.0)
    
class HableTonemapper(BaseTonemapper):
    def __init__(self, gain: float = 2.0):
        super(HableTonemapper, self).__init__()
        self.gain = gain

    def _tonemap_partial(self, x: torch.Tensor) -> torch.Tensor:
        A = 0.15
        B = 0.50
        C = 0.10
        D = 0.20
        E = 0.02
        F = 0.30

        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._tonemap_partial(x * self.gain)

        w = 11.2
        white_scale = 1.0 / self._tonemap_partial(w)
        res =  x * white_scale
        return res.clamp(0.0, 1.0)
    
class SRGBHable(BaseTonemapper):
    def __init__(self, gain: float = 2.0):
        super(SRGBHable, self).__init__()
        self.habel = HableTonemapper(gain)
        self.gamma = 2.2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.pow(1.0 / self.gamma)
        x = self.habel(x)
        return x.clamp(0.0, 1.0)
    
class SRGBTonemapper(BaseTonemapper):
    def __init__(self, gain: float = 2.0):
        super(SRGBTonemapper, self).__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gain
        x = x.pow(1.0 / 2.2)
        return x.clamp(0.0, 1.0)
    
class Exp(BaseTonemapper):
    def __init__(self, gain: float = 1.0):
        super(Exp, self).__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gain
        res = 1.0 - torch.exp(-x)
        return res.clamp(0.0, 1.0)


class Log(BaseTonemapper):
    def __init__(self, gain: float = 1.0):
        super(Log, self).__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gain
        res = torch.log(1.0 + x)
        return res.clamp(0.0, 1.0)
    
class DefaultTonemapper(BaseTonemapper):
    def __init__(self, gain:float = 1.0):
        super(DefaultTonemapper, self).__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.gain
        return x.clamp(0.0, 1.0)
