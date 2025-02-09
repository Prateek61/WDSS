import torch
import torch.nn as nn

from .relu import ReLUINR
from .wire import WIRE2D
from .fourier_mapping import FourierMapping, SimpleMapping
from .wire import WIRE2D    
from .siren import Siren
from .bwspline import BWSpline

from typing import List, Dict, Any

def get_fminr(config: Dict[str, Any]) -> nn.Module:
    if config['name'] == 'FMINR' and config['version'] == 1.0:
        return FMINR.from_config(config)
    else:
        assert False, f"Unknown config: {config}"

def get_inr(config: Dict[str, Any]) -> nn.Module:
    if config['type'] == 'relu' and config['version'] == 1.0:
        return ReLUINR.from_config(config)
    elif config['type'] == 'wire' and config['version'] == 1.0:
        return WIRE2D.from_config(config)
    elif config['type'] == 'siren' and config['version'] == 1.0:
        return Siren.from_config(config)
    elif config['type'] == 'bspline' and config['version'] == 1.0:
        return BWSpline.from_config(config)
    else:
        assert False, f"Unknown config: {config}"
    
class FMINR(nn.Module):
    def __init__(
        self,
        lr_feat_c: int,
        gb_feat_c: int,
        out_c: int,
        mlp_inp_c: int,
        mlp_config: Dict[str, Any] = {
            'type': 'relu',
            'version': 1.0,
            'in_channels': 64,
            'out_channels': 64,
            'num_layers': 4,
            'layer_size': 64
        },
        fourier_mapped: bool = False
    ):
        super(FMINR, self).__init__()

        self.lr_feat_c = lr_feat_c
        self.gb_feat_c = gb_feat_c
        self.out_c = out_c
        self.mlp_inp_c = mlp_inp_c
        self.fourier_mapped = fourier_mapped
        self.mlp_config = mlp_config
        self.fourier_mapped = fourier_mapped

        if self.fourier_mapped:
            self.mapping = FourierMapping(self.lr_feat_c, self.gb_feat_c, self.mlp_inp_c)
        else:
            self.mapping = SimpleMapping(self.lr_feat_c, self.gb_feat_c, self.mlp_inp_c)

        self.mlp = get_inr(self.mlp_config)

    def forward(self, lr_feat: torch.Tensor, gb_feat: torch.Tensor, upscale_factor: float) -> torch.Tensor:
        mapped_feat = self.mapping(lr_feat, gb_feat, upscale_factor)
        return self.mlp(mapped_feat)
    
    @staticmethod
    def from_config(config: Dict[str, Any]) -> 'FMINR':
        return FMINR(config['lr_feat_c'], config['gb_feat_c'], config['out_c'], config['mlp_inp_c'], config['mlp_config'], config['fourier_mapped'])
    